import os
import numpy as np
import imageio
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
import re
from typing import Sequence, Optional, Tuple

class NoisyBlockObsWrapper:
    """
    Injects noise into cube position observation and ensures camera images are in obs.
    - obs['cube_pos_noisy']: noisy cube position (agent-visible)
    - info['cube_pos_true']: true cube position (for logging/eval only)
    - obs['<camera>_image']: RGB image (H, W, 3), uint8
    """
    def __init__(
        self,
        env,
        add_noise: bool = True,
        expose_true_in_obs: bool = False,
        camera_names: Sequence[str] = ("frontview",),
        image_size: Sequence[int] = (128, 128),
        rng: Optional[np.random.RandomState] = None,
        flip_obs_images: bool = True,   # flip obs images upright if needed
    ):
        self.env = env
        self.expose_true_in_obs = bool(expose_true_in_obs)
        self.camera_names = list(camera_names)
        self.image_size = tuple(image_size)
        self.rng = rng or np.random.RandomState()
        self.flip_obs_images = bool(flip_obs_images)

        # cache cube body id
        self._cube_body = "cube_main"
        self._cube_bid = self.env.sim.model.body_name2id(self._cube_body)

        self.col_geom  = "cube_g0"       # collision geom
        self.vis_geom  = "cube_g0_vis"   # visual geom (may have material)

        self.m, self.d = env.sim.model, env.sim.data
        self.bid = self.m.body_name2id(self._cube_body)
        self.gid_col = self.m.geom_name2id(self.col_geom)
        self.gid_vis = self.m.geom_name2id(self.vis_geom)

        # Size (half-extent)
        half = 0.05                  # 0.05 cm -> 10 cm cube
        self.m.geom_size[self.gid_col] = np.array([half, half, half], dtype=float)
        self.m.geom_size[self.gid_vis] = np.array([half, half, half], dtype=float)

        obs = self.env.reset()
        self.true_pos = self.env.sim.data.body_xpos[self._cube_bid].copy().astype(np.float32)
        if self.expose_true_in_obs:
            obs["cube_pos_true"] = self.true_pos
        self._last_true_pos = self.true_pos

        noise = np.zeros(3, dtype=float)
        if add_noise:
            noise[1] = half*self.rng.uniform(-2.0, 2.0)

        obs["cube_pos_noisy"] = (self.true_pos + noise).astype(np.float32)

        self.noise = noise


    # image flip
    def image_flip(self, img: np.ndarray) -> np.ndarray:
        return img[::-1, :, :] if self.flip_obs_images else img

    # external getter (noisy or true cube position)
    def get_cube_pos(self, noisy: bool = True) -> np.ndarray:
        if not noisy:
            return self.true_pos
        return self.true_pos + self.noise

    # Gym-style API: reset
    def reset(self):
        obs = self.env.reset()

        # inject noisy cube position into obs
        true_pos = self.env.sim.data.body_xpos[self._cube_bid].copy().astype(np.float32)
        obs["cube_pos_noisy"] = (true_pos + self.noise).astype(np.float32)

        # optional: expose true position in obs
        if self.expose_true_in_obs:
            obs["cube_pos_true"] = true_pos
        self._last_true_pos = true_pos
        

        # ensure camera images exist and orientation is consistent
        for cam in self.camera_names:
            key = f"{cam}_image"
            if key in obs:
                obs[key] = self.image_flip(obs[key])
            else:
                w, h = int(self.image_size[0]), int(self.image_size[1])
                raw = self.env.sim.render(w, h, camera_name=cam)
                obs[key] = self.image_flip(raw)

        return obs

    # Gym-style API: step
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # inject noisy cube position into obs
        true_pos = self.env.sim.data.body_xpos[self._cube_bid].copy().astype(np.float32)
        obs["cube_pos_noisy"] = (true_pos + self.noise).astype(np.float32)

        # optional: expose true position in obs
        if self.expose_true_in_obs:
            obs["cube_pos_true"] = true_pos

        # put true cube position into info for logging/eval
        info = dict(info)
        info["cube_pos_true"] = true_pos

        # ensure camera images exist and orientation is consistent
        for cam in self.camera_names:
            key = f"{cam}_image"
            if key in obs:
                obs[key] = self.image_flip(obs[key])
            else:
                w, h = int(self.image_size[0]), int(self.image_size[1])
                raw = self.env.sim.render(w, h, camera_name=cam)
                obs[key] = self.image_flip(raw)

        return obs, reward, done, info

    # passthroughs
    @property
    def action_dim(self):
        return self.env.action_dim

    @property
    def sim(self):
        return self.env.sim

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def make_noisy_lift_env(
    add_noise: bool = True,
    camera_names: Sequence[str] = ("frontview",),
    image_size: Sequence[int] = (128, 128),   # (W, H)
    controller_name: str = "BASIC",
    flip_obs_images: bool = True,             # flip obs images upright if needed
):
    """
    Builds Lift env with noisy cube position and camera observations.
    """
    # offscreen rendering for Colab/MuJoCo
    os.environ.setdefault("MUJOCO_GL", "egl")

    controller_config = load_composite_controller_config(controller_name, robot="Panda")

    # base env with camera observations
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(camera_names),
        camera_heights=int(image_size[1]),
        camera_widths=int(image_size[0]),
        camera_depths=False,
        render_camera=camera_names[0],
        controller_configs=controller_config,
    )

    # wrap with noisy obs + image handling
    env = NoisyBlockObsWrapper(
        env,
        add_noise=add_noise,
        expose_true_in_obs=False,
        camera_names=camera_names,
        image_size=image_size,
        flip_obs_images=flip_obs_images,
    )
    return env


def init_frames_dir(frames_dir: str = "frames"):
    """
    Create (if needed) and clear the directory used to store image frames.
    All *.png files inside this directory will be removed.
    """
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(frames_dir, f))


def save_frame(obs, cam_key: str, frame_id: int, frames_dir: str = "frames") -> int:
    """
    Save the current camera image from `obs[cam_key]` as a PNG file.

    Args:
        obs: environment observation dict containing the camera image.
        cam_key: key of the camera image in `obs` (e.g., "frontview_image").
        frame_id: current frame index (used for filename).
        frames_dir: directory where frames are saved.

    Returns:
        Updated frame_id (incremented by 1).
    """
    img = obs[cam_key]
    imageio.imwrite(
        os.path.join(frames_dir, f"frame_{frame_id:04d}.png"),
        np.asarray(img, dtype=np.uint8),
    )
    return frame_id + 1


def step_with_action(
    env,
    action: np.ndarray,
    n_steps: int,
    obs: dict,
    cam_key: str,
    frame_id: int,
    frames_dir: str = "frames",
):
    """
    Repeat the same action for `n_steps` steps and save a frame at each step.

    Args:
        env: robosuite environment.
        action: action vector to be applied each step.
        n_steps: number of steps to repeat the action.
        obs: current observation (will be updated).
        cam_key: key for the camera image in obs.
        frame_id: current frame index.
        frames_dir: directory where frames are saved.

    Returns:
        (obs, frame_id):
            obs      - updated observation after the last step
            frame_id - updated frame index after saving all frames
    """
    for _ in range(n_steps):
        obs, _, _, _ = env.step(action)
        frame_id = save_frame(obs, cam_key, frame_id, frames_dir)
    return obs, frame_id


# ---------- Cartesian PID helpers ----------

def _to_3vec(x):
    """
    Convert a scalar or length-3 iterable into a np.ndarray of shape (3,).

    This is used so that kp/ki/kd can be given either as a scalar
    (same gain for x, y, z) or as a 3D vector.
    """
    if np.isscalar(x):
        return np.array([x, x, x], dtype=float)
    arr = np.array(x, dtype=float).reshape(-1)
    assert arr.shape[0] == 3, "kp/ki/kd must be scalar or length-3"
    return arr


def _get_dt_default(env, default=1.0 / 20.0):
    """
    Try to infer the control timestep (dt) from the environment.

    It first looks for `env.control_timestep`, then for
    `env.robots[0].control_timestep`. If nothing is found, it
    falls back to the given default value.

    Returns:
        dt as a float (seconds).
    """
    dt = getattr(env, "control_timestep", None)
    if dt is None and hasattr(env, "robots") and len(env.robots) > 0:
        dt = getattr(env.robots[0], "control_timestep", None)
    try:
        return float(dt) if dt is not None else float(default)
    except Exception:
        return float(default)


def move_ee_to(
    env,
    obs: dict,
    target_pos_or_fn,
    gripper: float,
    steps: int,
    action_dim: int,
    cam_key: str,
    frame_id: int,
    frames_dir: str = "frames",
    max_delta: float = 0.05,
    kp=10.0,
    ki=0.0,
    kd=0.0,
    i_clamp: float = 0.10,
    dt=None,
):
    """
    Move the end-effector towards a target position using a Cartesian PID.

    The controller operates in position space:
      - Reads `obs["robot0_eef_pos"]`
      - Computes error to a target 3D position
      - Outputs position deltas as action[:3]
      - Keeps orientation fixed (action[3:6] = 0)
      - Writes the gripper command to action[-1]

    Args:
        env: robosuite environment.
        obs: current observation (will be updated inside the loop).
        target_pos_or_fn: either a fixed 3D position (array-like of length 3)
                          or a callable `fn(obs) -> 3D position`.
        gripper: gripper command (e.g., -1.0 open, +1.0 close depending
                 on your controller).
        steps: number of control steps to run.
        action_dim: dimension of the action space (env.action_dim).
        cam_key: key for the camera image in obs.
        frame_id: current frame index.
        frames_dir: directory where frames are saved.
        max_delta: maximum magnitude of position delta per step along each axis.
        kp, ki, kd: PID gains (scalar or length-3).
        i_clamp: absolute clamp for the integral term.
        dt: control timestep; if None, it will be inferred from the env.

    Returns:
        (obs, frame_id):
            obs      - updated observation after the last step
            frame_id - updated frame index after saving all frames
    """

    # Helper to unify fixed target vs callable target
    def _get_target(o):
        return target_pos_or_fn(o) if callable(target_pos_or_fn) else np.asarray(
            target_pos_or_fn, dtype=float
        )

    kp_vec = _to_3vec(kp)
    ki_vec = _to_3vec(ki)
    kd_vec = _to_3vec(kd)
    dt = _get_dt_default(env) if dt is None else float(dt)

    integ = np.zeros(3, dtype=float)
    prev_err = None

    # In your current usage, the target is fixed per phase:
    # we compute it once before the loop and keep it constant.
    tgt = _get_target(obs)

    for _ in range(steps):
        curr = obs["robot0_eef_pos"]
        err = tgt - curr

        # PID terms
        integ = np.clip(integ + err * dt, -i_clamp, i_clamp)
        derr = (err - prev_err) / dt if prev_err is not None else np.zeros(
            3, dtype=float
        )
        prev_err = err.copy()

        dpos = kp_vec * err + ki_vec * integ + kd_vec * derr
        dpos = np.clip(dpos, -max_delta, max_delta)

        action = np.zeros(action_dim, dtype=float)
        action[:3] = dpos
        action[-1] = gripper

        obs, _, _, _ = env.step(action)
        frame_id = save_frame(obs, cam_key, frame_id, frames_dir)

    return obs, frame_id

def is_lift_success(
    obs: dict,
    cube_start_pos: np.ndarray,
    min_lift: float = 0.20,
    max_xy_shift: float = 0.08,
    use_noisy: bool = True,
) -> bool:
    """
    Check whether the lift task is considered successful.

    The default criteria are:
      - The cube's height increased by at least `min_lift` meters.
      - The cube did not drift too far in the XY plane from its initial position.

    Args:
        obs:
            Current observation dict (after your scripted rollout).
        cube_start_pos:
            Initial cube position at the start of the episode (shape (3,)).
            Usually you pass the cube position right after env.reset().
        min_lift:
            Minimum vertical displacement (in meters) required for success.
        max_xy_shift:
            Maximum allowed Euclidean distance in the XY plane from the start
            position. This prevents counting a far-away cube as a success.
        use_noisy:
            If True, use "cube_pos_noisy" from the wrapper.
            If False, try "cube_pos_true" (if exposed), then fall back to "cube_pos".

    Returns:
        True if the lift is successful according to the criteria, False otherwise.
    """
    if use_noisy:
        key = "cube_pos_noisy"
    else:
        # Prefer true position if available, otherwise fall back to base key
        if "cube_pos_true" in obs:
            key = "cube_pos_true"
        else:
            key = "cube_pos"

    assert key in obs, f"{key} not found in obs. Available keys: {list(obs.keys())}"

    cube_pos = np.asarray(obs[key], dtype=float).copy()

    # Vertical displacement (z axis)
    dz = cube_pos[2] - cube_start_pos[2]

    # Horizontal drift in XY
    xy_shift = np.linalg.norm(cube_pos[:2] - cube_start_pos[:2])

    return (dz >= min_lift) and (xy_shift <= max_xy_shift)