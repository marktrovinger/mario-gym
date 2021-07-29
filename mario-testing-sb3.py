from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb

config = {"policy_type":"MlpPolicy", "total_timesteps": 50000}
run = wandb.init(
    project="mario",
    config=config,
    sync_tensorboard=True,
    monitor_gym = True,
    save_code = True
 )


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = Monitor(env)
    return env
    
env = DummyVecEnv([make_env])

env = VecVideoRecorder(env, "videos",
    record_video_trigger=lambda x: x % 2000 == 0, video_length=200)



model = PPO(config['policy_type'], env, verbose=1,
    tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config['total_timesteps'], 
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_freq=1000,
        model_save_path=f"models/{run.id}",
    ),
)
