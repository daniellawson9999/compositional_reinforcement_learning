import gym
import d4rl
import time
from mujoco_py.generated import const


# Playing with camera: https://github.com/openai/mujoco-py/issues/10
env = gym.make('antmaze-large-play-v0')
import pdb; pdb.set_trace()

env.reset()
env.render()

env.viewer.cam.elevation = -90
env.viewer.cam.distance = env.model.stat.extent
#env.viewer.cam.type = const.CAMERA_FIXED

while True:
    env.reset()
    env.render()
    for i in range(100):
        env.step(env.action_space.sample())
        env.render()
