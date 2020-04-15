from rainbow import *
import gym
from gym.wrappers import Monitor


def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env


# Select an environment
env_name = ["CartPole-v0", "MsPacman-v0"]
env_idx = 0  # YOU CAN CHANGE THIS
env = wrap_env(gym.make(env_name[env_idx]))

num_frames = 1000000
memory_size = 10000
batch_size = 32
target_update = 1000
frame_interval = 1000
plot = True
model_name = env_name[env_idx] + "_" + str(num_frames)
test = True
train = True

preprocess_function = None
if env_idx == 1:
    preprocess_function = preprocess_obs_pacman

agent = DQNAgent(env, memory_size, batch_size, target_update,
                 plot=plot,
                 frame_interval=frame_interval,
                 frame_preprocess=preprocess_function,
                 n_frames_stack=1,
                 model_name=model_name,
                 training_delay=num_frames // 100,
                 )
"""agent = DQNAgent(env, memory_size, batch_size, target_update,
                         no_dueling=True, no_categorical=True, no_double=True,
                         no_n_step=True, no_noise=True, no_priority=True,
                         plot=plot, frame_interval=frame_interval)"""
if train:
    score, loss = agent.train(num_frames)
    agent.save()

if test:
    agent.load()
    tot_score = 0
    for i in range(100):
        score, _ = agent.test()
        tot_score += score
    print("Average score:", tot_score / 100)
