import pygame
from pyvirtualdisplay import Display
display = Display(visible=0, size=(640, 480))
display.start()
import os
import gym
# os.environ['SDL_VIDEODRIVER'] = 'dummy'
# soln: conda install -c conda-forge ffmpeg

env = gym.make('Pendulum-v1')
env = gym.wrappers.RecordVideo(env, f"test_video/test")
env.reset()

done = False
while not done:
    _, _, done, _ = env.step(env.action_space.sample())
env.close()

