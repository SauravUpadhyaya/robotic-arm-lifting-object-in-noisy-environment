import os
import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict

# import local utils (utils.py lives in the project root)
import utils as up_utils  # type: ignore


class RobotLiftEnv(gym.Env):
    """Gym wrapper around the provided noisy Lift environment.

    Two observation modes are supported:
      - 'dict'  : returns {'image': C,H,W uint8, 'cube_pos_noisy': (3,) float32}
      - 'flatten': returns a single 1D float32 array [image_flat, cube_pos]

    Action: 3-delta to add to the noisy cube position.
    """

    def __init__(
        self,
        image_size=(128, 128),
        steps_per_action: int = 8,
        max_delta: float = 0.08,
        camera_names=("frontview",),
        frames_dir: str = "frames",
        obs_type: str = "dict",
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.steps_per_action = int(steps_per_action)
        self.max_delta = float(max_delta)
        self.camera_key = f"{camera_names[0]}_image"
        self.frames_dir = frames_dir
        self.obs_type = str(obs_type)

        # ensure frames directory exists so save_frame calls don't fail
        try:
            up_utils.init_frames_dir(self.frames_dir)
        except Exception:
            # best-effort: if init fails, attempt to create directory directly
            try:
                os.makedirs(self.frames_dir, exist_ok=True)
            except Exception:
                pass

        # underlying robosuite env wrapper
        self._env = up_utils.make_noisy_lift_env(
            add_noise=True, camera_names=camera_names, image_size=image_size
        )

        # obtain a sample obs to set spaces
        obs = self._env.reset()
        img = obs[self.camera_key]
        H, W, C = img.shape

        if self.obs_type == "dict":
            # image space: float32 channel-first normalized to [0,1]
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(low=0.0, high=1.0, shape=(C, H, W), dtype=np.float32),
                    "cube_pos_noisy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                }
            )
        else:
            flat_size = int(C * H * W + 3)
            # use float32 for flattened array
            self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(flat_size,), dtype=np.float32)

        # action is delta to add to measured cube position
        self.action_space = spaces.Box(low=-self.max_delta, high=self.max_delta, shape=(3,), dtype=np.float32)

        # internal state
        self._start_cube_pos = obs["cube_pos_noisy"].astype(np.float32).copy()
        self._last_obs = obs
        self._frame_id = 0

    def seed(self, seed: int = None):
        """Seed internal RNGs and the underlying environment.

        Returns the seed used.
        """
        if seed is None:
            seed = np.random.randint(0, 2 ** 31 - 1)
        self.np_random = np.random.RandomState(int(seed))
        try:
            setattr(self._env, "rng", self.np_random)
        except Exception:
            pass
        try:
            if hasattr(self._env, "seed"):
                self._env.seed(int(seed))
        except Exception:
            pass
        return [int(seed)]

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Accepts the newer Gym signature reset(seed=..., options=...).
        """
        if seed is not None:
            self.seed(seed)
        obs = self._env.reset()
        self._start_cube_pos = obs["cube_pos_noisy"].astype(np.float32).copy()
        self._last_obs = obs
        # track previous cube position to compute per-step delta rewards (helps critic)
        self._prev_cube_pos = self._start_cube_pos.copy()
        self._frame_id = 0
        # Return Gym v0.26+ style (obs, info) tuple to be compatible with VecEnv
        return self._format_obs(obs), {}

    def step(self, action):
        # action: delta to add to noisy cube position
        action = np.asarray(action, dtype=float).reshape(-1)[:3]
        action = np.clip(action, -self.max_delta, self.max_delta)

        obs = self._last_obs
        noisy = np.asarray(obs["cube_pos_noisy"], dtype=float)
        target = noisy + action

        try:
            obs, self._frame_id = up_utils.move_ee_to(
                self._env,
                obs,
                target,
                gripper=-1.0,
                steps=self.steps_per_action,
                action_dim=self._env.action_dim,
                cam_key=self.camera_key,
                frame_id=self._frame_id,
                frames_dir=self.frames_dir,
                max_delta=0.05,
            )
        except ValueError as e:
            # Defensive: if the inner robosuite env raises because an action was
            # attempted after termination (common when doing multi-step PID
            # writes inside a single outer step), return a terminated signal
            # so the VecEnv worker can handle the episode lifecycle instead of
            # crashing the subprocess.
            msg = str(e)
            if "terminated" in msg or "terminated episode" in msg or "executing action in terminated" in msg:
                # mark episode as terminated and return last known observation
                obs = self._last_obs
                self._frame_id = self._frame_id
                self._last_obs = obs
                # small negative reward to discourage hitting terminal unexpectedly
                reward = 0.0
                terminated = True
                truncated = False
                info = {"success": False, "error": "terminated_during_internal_step"}
                return self._format_obs(obs), float(reward), terminated, truncated, info
            # otherwise, re-raise unexpected ValueErrors
            raise

        self._last_obs = obs

        cube_pos = np.asarray(obs["cube_pos_noisy"], dtype=float)

        # Per-step deltas (preferable for bootstrapping and lower variance)
        prev = getattr(self, "_prev_cube_pos", self._start_cube_pos)
        delta_z = float(cube_pos[2] - prev[2])
        delta_xy = float(np.linalg.norm(cube_pos[:2] - prev[:2]))

        # Shaped per-step reward: encourage upward motion, penalize horizontal drift
        # These coefficients are tunable: upward gain larger than xy penalty.
        reward = 1.0 * delta_z - 0.3 * delta_xy

        # small time penalty to encourage faster lifts (optional small effect)
        reward -= 0.001

        # update prev for next step
        self._prev_cube_pos = cube_pos.copy()

        # Gym v0.26+ API: return (obs, reward, terminated, truncated, info)
        success = up_utils.is_lift_success(obs, self._start_cube_pos, use_noisy=True)
        if success:
            reward += 50.0

        terminated = bool(success)
        truncated = False
        info = {"success": bool(success)}

        return self._format_obs(obs), float(reward), terminated, truncated, info

    def _format_obs(self, obs: dict):
        # convert H,W,C uint8 -> C,H,W uint8
        img = np.asarray(obs[self.camera_key], dtype=np.uint8)
        img_ch_first = np.transpose(img, (2, 0, 1)).copy()
        if self.obs_type == "dict":
            # normalize image to [0,1]
            img_float = img_ch_first.astype(np.float32) / 255.0
            return {"image": img_float, "cube_pos_noisy": np.asarray(obs["cube_pos_noisy"], dtype=np.float32)}
        img_flat = img_ch_first.ravel().astype(np.float32)
        cube_v = np.asarray(obs["cube_pos_noisy"], dtype=np.float32).ravel()
        return np.concatenate([img_flat, cube_v], axis=0)

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass