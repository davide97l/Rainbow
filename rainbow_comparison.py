from rainbow import *
import gym
import torch
import matplotlib.pyplot as plt
import argparse


def set_seed(seed, env):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)


if __name__ == '__main__':
    # python rainbow_comparison.py --num_frames 2000 --plotting_interval 100
    ap = argparse.ArgumentParser()
    ap.add_argument("-nf", "--num_frames", type=int, default=2000,
                    help="number of training frames")
    ap.add_argument("-plt", "--plot", default=False, action='store_true',
                    help="Plot training stats during training for each network")
    ap.add_argument("-pi", "--plotting_interval", type=int, default=100,
                    help="Number of steps per plots update")
    args = ap.parse_args()

    # hyper parameters
    num_frames = args.num_frames
    memory_size = args.num_frames / 10
    batch_size = 32
    target_update = args.num_frames / 10
    plotting_interval = args.plotting_interval
    plot = args.plot
    # seed
    seed = 777

    # make environment
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    set_seed(seed, env)

    # train
    agent_rainbow = DQNAgent(env, memory_size, batch_size, target_update,
                             no_dueling=False, no_categorical=False, no_double=False,
                             no_n_step=False, no_noise=False, no_priority=False,
                             plot=plot, frame_interval=plotting_interval)
    agent_rainbow_4 = DQNAgent(env, memory_size, batch_size, target_update,
                               no_dueling=False, no_categorical=False, no_double=False,
                               no_n_step=False, no_noise=False, no_priority=False,
                               plot=plot, frame_interval=plotting_interval, n_frames_stack=4)
    agents = [agent_rainbow, agent_rainbow_4]

    labels = ["Rainbow", "Rainbow-4-frames"]

    scores = []
    losses = []
    for i, agent in enumerate(agents):
        print("Training agent", labels[i])
        score, loss = agent.train(num_frames)
        scores.append(score)
        losses.append(loss)

    # create a color palette
    palette = plt.get_cmap('Set1')

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('Training frames: %s' % num_frames)
    for i in range(len(scores)):
        linewidth = 3.
        plt.plot(scores[i], marker='', color=palette(i), linewidth=linewidth, alpha=1., label=labels[i])
    plt.legend(loc=2, ncol=1)
    plt.xlabel("Frames x " + str(plotting_interval))
    plt.ylabel("Score")
