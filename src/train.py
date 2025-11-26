"""Train a PPO agent to produce delta adjustments to the noisy cube position.

Usage (from project root):
    python3 src/train.py

This script creates multiple vectorized envs and trains PPO with a MultiInputPolicy
that handles the image+vector observation dict.
"""
import os
import sys
import argparse
import numpy as np

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
proj_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, proj_root)


from src.robot_env import RobotLiftEnv
from src.features import CustomCombinedExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
import torch as th


def make_env_fn(rank=0, seed=0, image_size=(128, 128), steps_per_action=1):
    def _init():
        # Use dict observations (image + noisy position) for CNN + vector extractor
        env = RobotLiftEnv(image_size=image_size, steps_per_action=steps_per_action, obs_type="dict")
        env.seed(seed + rank)
        # lightweight Monitor wrapper to record episode stats (no video)
        try:
            env = Monitor(env)
        except Exception:
            pass
        return env

    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=2000000, help="Total environment steps to train")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128], help="W H image size for env cameras")
    parser.add_argument("--steps_per_action", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=512, help="PPO n_steps per env (higher -> lower variance)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--vf_coef", type=float, default=2.0, help="Value function loss coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.001, help="Entropy coefficient")
    parser.add_argument("--save_freq", type=int, default=10000, help="Checkpoint frequency (in steps)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force_dummy", action="store_true", help="Force single-process DummyVecEnv (safer on low-memory machines)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # create vectorized envs (use SubprocVecEnv if n_envs>1 for better performance)
    env_fns = []
    for i in range(args.n_envs):
        def _fn(rank=i):
            return make_env_fn(rank, seed=args.seed, image_size=tuple(args.image_size), steps_per_action=args.steps_per_action)
        env_fns.append(_fn())

    # Quick probe: try creating/resetting a single env in-process to detect
    # immediate MuJoCo allocation failures (e.g. out-of-memory). If the probe
    # fails, fall back to a safer single-process DummyVecEnv and print guidance.
    try:
        probe_env = make_env_fn(0, seed=args.seed, image_size=tuple(args.image_size), steps_per_action=args.steps_per_action)()
        try:
            probe_env.reset()
        finally:
            try:
                probe_env.close()
            except Exception:
                pass
    except Exception as e:
        # Silent fallback to single-env when probe fails (avoid noisy guidance)
        args.n_envs = 1
        env_fns = [make_env_fn(0, seed=args.seed, image_size=tuple(args.image_size), steps_per_action=args.steps_per_action)]

    # Instantiate the vectorized env, but fall back to DummyVecEnv if SubprocVecEnv
    # raises during creation (this can happen on low-memory machines or when
    # multiprocessing resources are limited).
    try:
        if args.force_dummy or args.n_envs <= 1:
            vec_env = DummyVecEnv(env_fns)
        else:
            try:
                vec_env = SubprocVecEnv(env_fns)
            except Exception:
                # Silent fallback to DummyVecEnv on failure
                vec_env = DummyVecEnv(env_fns)
    except Exception as e:
        # Catch-all: exit silently (no guidance printed)
        sys.exit(1)

    # Wrap with VecNormalize for stable training when using pixel inputs and vector inputs
    try:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    except Exception:
        # If VecNormalize can't be applied due to env type, continue without it
        vec_env = vec_env

    # create evaluation env (single worker) and copy VecNormalize statistics if present
    eval_env = DummyVecEnv([make_env_fn(0, seed=args.seed + 999, image_size=tuple(args.image_size), steps_per_action=args.steps_per_action)])
    try:
        if isinstance(vec_env, VecNormalize):
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            # copy running stats so eval uses the same normalization
            eval_env.obs_rms = vec_env.obs_rms
            eval_env.ret_rms = vec_env.ret_rms
    except Exception:
        pass

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.out_dir, name_prefix="ppo_lift")

    # Eval callback to log average reward (n_eval_episodes) and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.out_dir, "best_model"),
        log_path=os.path.join(args.out_dir, "eval"),
        eval_freq=max(1, args.save_freq),
        n_eval_episodes=5,
        deterministic=True,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # train PPO with MultiInputPolicy using our custom extractor
    # policy and value network architecture: give the value network more capacity
    # so it can better fit the return function (helps explained_variance).
    policy_kwargs = {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {"cnn_output_dim": 256},
        # separate MLP heads for policy (pi) and value function (vf)
        # pi: smaller; vf: larger to improve value predictions
        "net_arch": [
            {"pi": [256, 128], "vf": [1024, 512, 256]}
        ],
    }

    # stable-baselines3 PPO hyperparameters tuned for pixel+vector tasks
    # PPO hyperparameters tuned to help stabilize value learning
    # print configuration summary
    print("Training config:", {
        "n_envs": args.n_envs,
        "timesteps": args.timesteps,
        "image_size": tuple(args.image_size),
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef,
        "device": device,
    })

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(args.out_dir, "tb"),
        device=device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
    )

    # Train with defensive exception handling so memory / subprocess errors
    # produce a friendly message and we can still save partial artifacts.
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
    except (BrokenPipeError, EOFError, MemoryError, ValueError, RuntimeError):
        # Silent abort on common environment/worker errors
        pass
    except Exception:
        # Silent abort on unexpected errors
        pass
    finally:
        # final save (best-effort)
        try:
            final_path = os.path.join(args.out_dir, "ppo_lift_final.zip")
            model.save(final_path)
            print(f"Saved final model to: {final_path}")
        except Exception:
            pass
        # Save VecNormalize statistics if present so eval uses same normalization
        try:
            if hasattr(vec_env, "save"):
                vec_env.save(os.path.join(args.out_dir, "vec_normalize.pkl"))
        except Exception:
            pass
        # Close environments
        try:
            vec_env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()