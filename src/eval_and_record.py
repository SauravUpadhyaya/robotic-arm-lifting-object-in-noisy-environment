"""Evaluate a trained policy on 10 randomized environments and record videos.

Usage (from project root):
    python3 src/eval_and_record.py --model models/ppo_lift_final.zip

The script will print success rate and write frames for the best run to ./records.
"""
import os
import sys
import argparse
import numpy as np
import warnings
import logging

proj_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, proj_root)

# suppress common noisy logs before any heavy imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'osmesa')  # optional: offscreen backend
import warnings
warnings.filterwarnings("ignore")  # coarse; optional more selective filters below

# Python logging settings
import logging
logging.getLogger('robosuite').setLevel(logging.ERROR)
logging.getLogger('gym').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from importlib import import_module


# Try importing the robot env while being resilient to an invalid MUJOCO_GL
# value (e.g. 'osmesa' not available on this system). We try the current
# value, then common alternatives, then unset the variable.
def _import_robot_env_with_fallbacks():
    tried = set()
    # order: user-specified, glfw, egl, unset
    candidates = [os.environ.get('MUJOCO_GL'), 'glfw', 'egl', None]
    last_err = None
    for cand in candidates:
        if cand in tried:
            continue
        tried.add(cand)
        if cand is None:
            if 'MUJOCO_GL' in os.environ:
                del os.environ['MUJOCO_GL']
            print('[info] MUJOCO_GL unset -> trying default backend')
        else:
            os.environ['MUJOCO_GL'] = cand
            print(f'[info] trying MUJOCO_GL={cand}')
        try:
            mod = import_module('src.robot_env')
            # reload to ensure it picks up the env var we set
            mod = import_module('src.robot_env')
            return mod.RobotLiftEnv
        except RuntimeError as e:
            last_err = e
            msg = str(e)
            # if the error is because of MUJOCO_GL, try next candidate
            if 'invalid value for environment variable MUJOCO_GL' in msg or 'MUJOCO_GL' in msg:
                continue
            # otherwise re-raise
            raise
    # If we get here, raise the last RuntimeError so the user can see details
    raise last_err or RuntimeError('failed to import src.robot_env')


RobotLiftEnv = _import_robot_env_with_fallbacks()
from stable_baselines3 import PPO
import utils as local_utils  # type: ignore


def run_once(model, out_dir: str, record=False, image_size=(128, 128), obs_type="dict", steps_per_action: int = 6,
             model_prefers: str = "dict", model_expected_image_shape=None, max_steps: int = 1000):
    # If quick mode (flatten observations) is requested, use smaller image and no frame recording
    env = RobotLiftEnv(image_size=tuple(image_size), steps_per_action=steps_per_action, frames_dir=out_dir, obs_type=obs_type)
    reset_res = env.reset()
    # reset may return (obs, info) tuple
    if isinstance(reset_res, tuple):
        obs, _ = reset_res
    else:
        obs = reset_res
    frame_id = 0
    if record:
        local_utils.init_frames_dir(out_dir)

    done = False
    total_reward = 0.0
    info = {}
    action_counter = 0
    # safety guard: avoid infinite/very long episodes; break after max_steps
    while not done:
        # Use precomputed model preference/expected image shape (faster than
        # querying the model.observation_space each loop).
        if isinstance(obs, dict) and (model_prefers == "dict") and (model_expected_image_shape is not None):
            target_shape = model_expected_image_shape  # (C, H, W)
            cur_img = obs["image"]
            if cur_img.shape != target_shape:
                # resize needed: convert C,H,W -> H,W,C for PIL/OpenCV
                C, H, W = cur_img.shape
                tgt_C, tgt_H, tgt_W = target_shape
                img_hwc = np.transpose(cur_img, (1, 2, 0))

                # helper: try cv2, then PIL, else nearest-neighbor repeat for integer scales
                resized_hwc = None
                try:
                    import cv2

                    # img_hwc is float in [0,1] or uint8; convert to uint8 for cv2
                    img_cv = (np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
                    resized = cv2.resize(img_cv, (tgt_W, tgt_H), interpolation=cv2.INTER_LINEAR)
                    resized_hwc = resized.astype(np.float32) / 255.0
                except Exception:
                    try:
                        from PIL import Image

                        img_pil = Image.fromarray((np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8))
                        img_pil = img_pil.resize((tgt_W, tgt_H), resample=Image.BILINEAR)
                        resized_hwc = (np.asarray(img_pil).astype(np.float32) / 255.0)
                    except Exception:
                        # fallback: integer upscale using repeat
                        if tgt_H % H == 0 and tgt_W % W == 0:
                            fy = tgt_H // H
                            fx = tgt_W // W
                            resized_hwc = np.repeat(np.repeat(img_hwc, fy, axis=0), fx, axis=1)
                        else:
                            # last resort: centre-crop/pad to target (not ideal)
                            resized_hwc = np.zeros((tgt_H, tgt_W, C), dtype=np.float32)
                            h = min(H, tgt_H)
                            w = min(W, tgt_W)
                            resized_hwc[:h, :w, :] = img_hwc[:h, :w, :]

                # convert back to C,H,W and assign into obs copy used for predict
                img_chw = np.transpose(resized_hwc, (2, 0, 1)).astype(np.float32)
                obs_for_pred = dict(obs)
                obs_for_pred["image"] = img_chw
            else:
                obs_for_pred = obs
        else:
            obs_for_pred = obs

        action, _ = model.predict(obs_for_pred, deterministic=True)
        step_res = env.step(action)
        # support both old (obs, r, done, info) and new (obs, r, terminated, truncated, info)
        if len(step_res) == 5:
            obs, r, terminated, truncated, info = step_res
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = step_res
        total_reward += r
        action_counter += 1

        # print light progress so the user sees activity (not too spammy)
        # avoid printing when action_counter==0 (0 % 20 == 0) to prevent misleading output
        if action_counter > 0 and (action_counter % 20) == 0:
            print(f"[eval] steps={action_counter} total_reward={total_reward:.2f}", flush=True)

        if action_counter >= int(max_steps):
            print(f"[eval] reached max_steps={max_steps}, aborting episode", flush=True)
            break
    success = info.get("success", False)

    # diagnostic: count saved frames in out_dir (if any) to help detect I/O stalls
    frames_saved = 0
    try:
        if os.path.isdir(out_dir):
            frames_saved = sum(1 for f in os.listdir(out_dir) if f.lower().endswith('.png'))
    except Exception:
        frames_saved = 0

    print(f"[eval] episode finished: agent_steps={action_counter} frames_saved={frames_saved} total_reward={total_reward:.2f} success={bool(success)}", flush=True)
    env.close()
    return bool(success), total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=str, default="records")
    parser.add_argument("--quick", action="store_true", help="Run a faster, lower-resolution evaluation (no frames saved unless --out provided)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128], help="Image size W H to use for the env cameras")
    parser.add_argument("--steps_per_action", type=int, default=6, help="Number of internal control steps run per agent action (lower is faster)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max agent steps per episode (safety cap)")
    args = parser.parse_args()

    model = PPO.load(args.model)

    # suppress noisy warnings and reduce robosuite log spam for faster runs
    warnings.filterwarnings("ignore")
    try:
        logging.getLogger("robosuite").setLevel(logging.ERROR)
    except Exception:
        pass

    # Determine whether the loaded model expects dict observations (MultiInputPolicy)
    try:
        obs_space = model.observation_space
    except Exception:
        obs_space = None
    if hasattr(obs_space, "spaces"):
        model_prefers = "dict"
    else:
        model_prefers = "flatten"

    # If the model expects dict obs with an 'image' entry, grab its expected shape
    model_expected_image_shape = None
    if model_prefers == "dict":
        try:
            model_expected_image_shape = obs_space.spaces["image"].shape
        except Exception:
            model_expected_image_shape = None

    os.makedirs(args.out, exist_ok=True)

    successes = 0
    best_score = -1e9
    best_idx = None

    for i in range(args.n):
        cur_dir = os.path.join(args.out, f"run_{i:02d}")
        os.makedirs(cur_dir, exist_ok=True)
        print(f"[eval] Starting run {i+1}/{args.n} -> {cur_dir}", flush=True)
        # Determine obs_type to use: prefer the model's expected format.
        if args.quick:
            # quick mode uses smaller images and disables frame saving. Keep obs format compatible with model.
            use_obs_type = model_prefers
            record = False
        else:
            use_obs_type = model_prefers
            record = True

        success, score = run_once(
            model,
            cur_dir,
            record=record,
            image_size=tuple(args.image_size),
            obs_type=use_obs_type,
            steps_per_action=args.steps_per_action,
            model_prefers=model_prefers,
            model_expected_image_shape=model_expected_image_shape,
            max_steps=args.max_steps,
        )
        print(f"Run {i}: success={success} score={score:.3f} (frames: {cur_dir})")
        if success:
            successes += 1
        if score > best_score:
            best_score = score
            best_idx = i

    print(f"Success rate: {successes}/{args.n} = {successes/args.n:.2f}")
    print(f"Best run: {best_idx} score={best_score:.3f}")


if __name__ == "__main__":
    main()
