import argparse
from typing import Optional

import gymnasium as gym
import numpy as np
import retro
import torch
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env import (
  SubprocVecEnv,
  VecFrameStack,
  VecEnv,
  VecEnvWrapper,
)
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from transformers import AutoImageProcessor, YolosForObjectDetection


class YolosWrapper(VecEnvWrapper):
  """Takes a frame stacked observation and applies the yolo model to the batch"""

  def __init__(self, venv: VecEnv, device: str):
    super().__init__(venv=venv)
    self.device = device
    self.image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
    self.image_processor.do_resize = False
    self.image_processor.do_pad = True
    self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
    self.model.eval()
    self.model.to(device)
    self.observation_space = gym.spaces.Box(low=0, high=1, shape=(231, 384), dtype=np.float32)

  def inference(self, obs: np.ndarray) -> np.ndarray:
    pixel_values = self.image_processor(obs, return_tensors="pt").pixel_values.to(self.device)
    with torch.no_grad():
      inference = self.model(pixel_values).last_hidden_state.squeeze().detach().to("cpu").numpy()
      return inference

  def reset(self) -> np.ndarray:
    return self.inference(self.venv.reset())

  def step_wait(self) -> VecEnvStepReturn:
    obs, rewards, dones, infos = self.venv.step_wait()
    return (
      self.inference(obs),
      rewards,
      dones,
      infos,
    )


def make_retro(
  *,
  game: str,
  roms_path: str,
  state: Optional[str] = None,
  max_episode_steps: int = 4500,
  **kwargs,
):
  retro.data.Integrations.add_custom_path(roms_path)
  if state is None:
    state = retro.State.DEFAULT
  env = retro.make(game, state, inttype=retro.data.Integrations.ALL, **kwargs)
  if max_episode_steps is not None:
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
  return env


def wrap_deepmind_retro(env: gym.Env):
  """
  Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
  """
  env = WarpFrame(env)
  env = ClipRewardEnv(env)
  return env


if __name__ == "__main__":
  # lets begin with testing atari games in stable retro and stable baselines
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "roms_path",
    help="Path to directory with directories of roms. "
    "Go to https://stable-retro.farama.org/integration/#using-a-custom-integration-from-python "
    "to learn how to format your roms directory.",
  )
  parser.add_argument("--device", choices=["cuda", "cpu"])
  parser.add_argument("--n-stack", type=int, help="Number of frames to stack", default=4)
  parser.add_argument("--render-mode", choices=["rgb_array", "human"], default="rgb_array")
  args = parser.parse_args()

  # Note the YolosWrapper could be post frame stacking. Not sure that will work for cheap GPUs
  # + it's annoying to think about the reshape math unless I make a fused wrapper
  # Probably should roll out my own custom env tbh...
  venv = VecFrameStack(
    YolosWrapper(
      SubprocVecEnv(
        [
          lambda: make_retro(
            game="Breakout-Atari2600",
            roms_path=args.roms_path,
            render_mode=args.render_mode,
            record=True,
          ),
          # lambda: make_retro(
          #     game="Qbert-Atari2600", roms_path=args.roms_path
          # ),
          # lambda: make_retro(
          #     game="SpaceInvaders-Atari2600", roms_path=args.roms_path
          # ),
        ]
        * 4
      ),
      device=args.device,
    ),
    n_stack=args.n_stack,
    channels_order="first",
  )
  model = PPO(
    policy="MlpPolicy",
    env=venv,
    learning_rate=lambda f: f * 2.5e-4,
    n_steps=512,
    batch_size=4,
    n_epochs=4096,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
    verbose=1,
    device=args.device,
  )
  model.learn(
    total_timesteps=100_000_000,
    log_interval=1,
  )
