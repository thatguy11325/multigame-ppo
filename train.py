import argparse
from typing import Optional

import gymnasium
import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame


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


def wrap_deepmind_retro(env: gymnasium.Env):
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
    args = parser.parse_args()

    venv = VecTransposeImage(
        VecFrameStack(
            SubprocVecEnv(
                [
                    lambda: make_retro(
                        game="Breakout-Atari2600", roms_path=args.roms_path
                    ),
                    lambda: make_retro(
                        game="Qbert-Atari2600", roms_path=args.roms_path
                    ),
                    lambda: make_retro(
                        game="SpaceInvaders-Atari2600", roms_path=args.roms_path
                    ),
                ]
            ),
            n_stack=4,
        )
    )
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=256,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
    )
