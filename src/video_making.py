"""Create an MP4 from saved frames using the fastest available backend.

This module provides a simple CLI for creating MP4s from recorded PNG
frames saved by the evaluation pipeline. It prefers the FFmpeg-backed
writer (imageio-ffmpeg or system ffmpeg) and falls back to imageio's
default writer if not available.

Usage examples:
  python3 src/video_making.py --input-dir records --best --out demo.mp4 --fps 20
  python3 src/video_making.py --input-dir records/run_03 --run 3 --out demo_run3.mp4

Dependencies:
  - imageio (required)
  - imageio-ffmpeg (recommended for speed) or system ffmpeg installed
    (brew install ffmpeg)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import imageio


def _choose_run_dir(input_dir: str, run_idx: Optional[int], pick_best: bool) -> str:
    """Return a directory containing frame_*.png images.

    The function supports three selection modes:
      - explicit run index: ``--run N`` -> ``input_dir/run_NN``
      - best: pick the run_* folder with the most frames (``--best``)
      - default: prefer ``run_00`` if present, otherwise the first run_* folder
        or the input_dir itself if it directly holds frames.
    """
    input_dir = os.fspath(input_dir)
    # Candidate run directories
    run_pattern = os.path.join(input_dir, "run_*")
    runs = sorted(d for d in glob.glob(run_pattern) if os.path.isdir(d))

    if run_idx is not None:
        candidate = os.path.join(input_dir, f"run_{int(run_idx):02d}")
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f"Requested run directory not found: {candidate}")

    if pick_best:
        best_dir: Optional[str] = None
        best_count = -1
        for d in runs:
            count = len(glob.glob(os.path.join(d, "frame_*.png")))
            if count > best_count:
                best_count = count
                best_dir = d
        if best_dir is None:
            raise FileNotFoundError(f"No run_* directories with frames found in {input_dir}")
        return best_dir

    # fallback: prefer run_00, then input_dir if it contains frames, then first run
    run00 = os.path.join(input_dir, "run_09")
    if os.path.isdir(run00):
        return run00
    if glob.glob(os.path.join(input_dir, "frame_*.png")):
        return input_dir
    if runs:
        return runs[0]
    raise FileNotFoundError(f"No runs or frames found in {input_dir}")


def _has_ffmpeg_backend() -> bool:
    """Return True if an FFmpeg-backed writer is available for imageio."""
    try:
        import imageio_ffmpeg  # type: ignore

        return True
    except Exception:
        # imageio.formats exists in v2 and v3; guard access
        try:
            return "ffmpeg" in imageio.formats
        except Exception:
            return False


def create_video_from_frames(run_dir: str, out_path: str, fps: int = 20) -> None:
    """Encode frames in ``run_dir`` to ``out_path`` at ``fps``.

    This function reads files matching `frame_*.png` in lexical order.
    """
    frames = sorted(glob.glob(os.path.join(run_dir, "frame_*.png")))
    if not frames:
        raise FileNotFoundError(f"No frames (frame_*.png) found in {run_dir}")

    use_ffmpeg = _has_ffmpeg_backend()
    if use_ffmpeg:
        print(f"[info] using ffmpeg backend for {out_path} (from {run_dir}, {len(frames)} frames)")
        writer = imageio.get_writer(out_path, fps=fps, format="ffmpeg", codec="libx264")
    else:
        print(f"[warning] ffmpeg backend not available; using default writer (slower)")
        writer = imageio.get_writer(out_path, fps=fps)

    try:
        for fp in frames:
            img = imageio.imread(fp)
            writer.append_data(img)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MP4 from recorded frames")
    parser.add_argument("--input-dir", default="records", help="Directory containing run_XX folders or frames")
    parser.add_argument("--run", type=int, help="Specific run index to use (e.g. 0 for run_00)")
    parser.add_argument("--best", action="store_true", help="Automatically pick the run with the most frames")
    parser.add_argument("--out", default="demo_best_run.mp4", help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=20, help="Output frames per second")
    args = parser.parse_args()

    try:
        run_dir = _choose_run_dir(args.input_dir, args.run, args.best)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(2)

    try:
        create_video_from_frames(run_dir, args.out, fps=args.fps)
    except Exception as e:  # show a helpful message
        print(f"[error] failed to create video: {e}")
        sys.exit(2)

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()