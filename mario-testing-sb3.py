from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation, FrameStack

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

done = True

model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=int(40000))
model.save("dqn_mario_downsample")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()