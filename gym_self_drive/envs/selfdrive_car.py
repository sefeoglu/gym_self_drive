import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SelfDrive(gym.Env):

    metadata = {'render.modes': ['human']}
    def __init__(self):
        super.__init__()
    def step(self, action):
        super.step()
    def reset(self):
        super.reset()
    def render(self):
        super.render()
    def close(self):
        super.close()
