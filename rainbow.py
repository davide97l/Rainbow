import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Deque, Dict, List, Tuple
from replay_buffer import *
from utils import *
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_


class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        n_step (int): step number to calculate n-step td error
        hidden_size (int): size of the layers of the top dense network
        lr (float): learning rate
        plot (bool): if True, plot training charts
        plotting_interval (int): steps per plotting refresh
        alpha (float): determines how much prioritization is used
        beta (float): determines how much importance sampling is used
        prior_eps (float): guarantees every transition can be sampled
        max_epsilon (float): starting epsilon value
        min_epsilon (float): finish epsilon value
        epsilon_decay (float): epsilon decay for each step
    """

    def __init__(
            self,
            env: gym.Env,
            memory_size: int = 1024,
            batch_size: int = 32,
            target_update: int = 100,
            gamma: float = 0.99,
            lr: float =0.001,
            hidden_size=128,
            # PER parameters
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            # Categorical DQN parameters
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 3,
            # Plotting
            plot: bool = True,
            plotting_interval: int = 100,
            # Options
            no_dueling: bool = False,
            no_double: bool = False,
            no_noise: bool = False,
            no_categorical: bool = False,
            no_priority: bool = False,
            no_n_step: bool = False,
            # Only if no_noise is True
            max_epsilon: float = 1.,
            min_epsilon: float = 0.1,
            epsilon_decay: float = 0.0005
    ):
        assert len(env.observation_space.shape) == 3 or len(env.observation_space.shape) == 1
        if len(env.observation_space.shape) == 1:
            obs_dim = [env.observation_space.shape[0]]
        else:
            # TODO convert to grayscale
            # remember: gym has dimension (w, h, c) but pytorch has (c, h, w)
            obs_dim = [env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2]]
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.hidden_size = hidden_size

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Device", self.device)

        # PER and memory for N-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, n_step=n_step, gamma=gamma
        )
        self.n_step = n_step

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        if len(obs_dim) == 1:
            # if input is 1d array use convolutional layers
            self.dqn = DenseNet(
                obs_dim[0], action_dim, self.atom_size, self.support, self.hidden_size, no_dueling, no_noise
            ).to(self.device)
            self.dqn_target = DenseNet(
                obs_dim[0], action_dim, self.atom_size, self.support, self.hidden_size, no_dueling, no_noise
            ).to(self.device)
        else:
            # if input is 3d frame use convolutional layers
            self.dqn = ConvNet(
                obs_dim, action_dim, self.atom_size, self.support, no_dueling, no_noise
            ).to(self.device)
            self.dqn_target = ConvNet(
                obs_dim, action_dim, self.atom_size, self.support, no_dueling, no_noise
            ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        # current transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # plot
        self.plot = plot
        self.plotting_interval = plotting_interval

        # epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = self.max_epsilon
        self.epsilon_decay = epsilon_decay

        # options to disable some features
        self.no_dueling = no_dueling
        self.no_double = no_double
        if no_double:
            self.dqn_target = self.dqn
        self.no_noise = no_noise
        if no_noise:
            self.epsilon, self.max_epsilon, self.min_epsilon = 0, 0, 0
        self.no_categorical = no_categorical
        self.no_n_step = no_n_step
        if no_n_step:
            self.n_step = 1
            self.memory = PrioritizedReplayBuffer(
                obs_dim, memory_size, batch_size, alpha=alpha, n_step=self.n_step, gamma=gamma
            )
        self.no_priority = no_priority
        if no_priority:
            self.alpha = 0
            self.memory = PrioritizedReplayBuffer(
                obs_dim, memory_size, batch_size, alpha=alpha, n_step=self.n_step, gamma=gamma
            )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state and return state and action."""
        # epsilon greedy policy
        if self.no_noise and self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            # Select best action: no epsilon greedy action selection but NoisyNet
            selected_action = self.dqn(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env (next state, rewards, done)."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # sample transitions
        samples = self.memory.sample_batch(self.beta)
        # PER needs beta to calculate weights
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # N-step Learning loss
        gamma = self.gamma ** self.n_step
        elementwise_loss = self._compute_dqn_loss(samples, gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        if not self.no_noise:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss.item()

    def train(self, num_frames: int) -> (List[int], List[int]):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        frame_scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # PER: increase beta (make weight sampling more effective)
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                scores.append(score)
                state = self.env.reset()
                score = 0

            # linearly decrease epsilon
            if self.no_noise:
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                # if hard update is needed
                if update_cnt % self.target_update == 0 and not self.no_double:
                    self._target_hard_update()

            # plotting
            if self.plot and frame_idx % self.plotting_interval == 0:
                if len(scores) == 0:
                    if len(frame_scores) > 0:
                        frame_scores.append(float(frame_scores[-1]))
                    else:
                        frame_scores.append(0.)
                else:
                    frame_scores.append(float(np.mean(scores)))
                self._plot(frame_idx, frame_scores, losses)
                scores = []

        self.env.close()

        return frame_scores, losses

    def test(self) -> int:
        """Test the agent on one episode."""
        self.is_test = True

        state = self.env.reset()
        done = False
        score = 0

        while not done:
            self.env.render()
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        return score

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return the loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        if not self.no_categorical:
            # # Compute categorical distribution loss
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

            with torch.no_grad():
                next_action = self.dqn(next_state).argmax(1)
                next_dist = self.dqn_target.dist(next_state)
                next_dist = next_dist[range(self.batch_size), next_action]

                t_z = reward + (1 - done) * gamma * self.support
                t_z = t_z.clamp(min=self.v_min, max=self.v_max)
                b = (t_z - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = (
                    torch.linspace(
                        0, (self.batch_size - 1) * self.atom_size, self.batch_size
                    ).long().unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device)
                )

                proj_dist = torch.zeros(next_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
                proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )

            dist = self.dqn.dist(state)
            log_p = torch.log(dist[range(self.batch_size), action])
            loss = -(proj_dist * log_p).sum(1)

        else:
            # Compute normal value estimation loss
            # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
            #       = r                       otherwise
            action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
            curr_q_value = self.dqn(state).gather(1, action)
            next_q_value = self.dqn_target(next_state).max(
                dim=1, keepdim=True
            )[0].detach()
            mask = 1 - done
            target = (reward + self.gamma * next_q_value * mask).to(self.device)

            # calculate dqn loss
            loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('Frame %s. Mean Score: %.4s' % (frame_idx,
                                                  np.mean(scores[-3:])))
        plt.plot(scores)
        plt.xlabel("Frames x " + str(self.plotting_interval))
        plt.ylabel("Score")
        plt.subplot(132)
        plt.title('Loss')
        plt.plot(losses)
        plt.xlabel("Frames")
        plt.ylabel("Loss")
        plt.show()
