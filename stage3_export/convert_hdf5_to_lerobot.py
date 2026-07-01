#!/usr/bin/env python3
"""
Stage 3 — HDF5 → LeRobot v2 converter.

Reads every episode_*.h5 produced by Stage 2 and writes a LeRobot v2 dataset
(parquet + MP4 + meta) ready to transfer to the lab server for GR00T
fine-tuning.

Run:
    python stage3_export/convert_hdf5_to_lerobot.py \
        --input-dir  /home/eduardot/gr00t_project/hdf5_episodes \
        --output-dir /home/eduardot/gr00t_project/handshake_dataset \
        --task "perform a handshake" \
        --fps 10 --camera-name head_cam

state_dim/action_dim are inferred from the first episode's actual
joint_pos/actions arrays, not hardcoded — this matches whatever DOF count
Stage 2 recorded (29 body + 22 Revo2 hand = 51 DOF) without needing to know it
in advance. An --expect-dof guard aborts the run if the recorded dim differs
(e.g. if pointed at the legacy 53-DOF Inspire backups).
"""

import argparse
import glob
import json
import os
import subprocess

import h5py
import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = "/home/eduardot/gr00t_project/configs/scene_config.yaml"


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def parse_args():
    cfg = load_config()
    rec = cfg.get("recording", {}) if cfg else {}
    p = argparse.ArgumentParser(description="Convert HDF5 episodes to LeRobot v2")
    p.add_argument("--input-dir", default="/mnt/data/gr00t_data/hdf5_episodes")
    p.add_argument("--output-dir", default="/mnt/data/gr00t_data/handshake_dataset")
    p.add_argument("--task", default="perform a handshake")
    p.add_argument("--fps", type=int, default=int(rec.get("fps", 10)))
    p.add_argument("--camera-name", default="head_cam")
    p.add_argument(
        "--action-horizon", type=int, default=int(rec.get("action_horizon", 40)),
        help="Number of future action steps (action delta_indices = range(N)) GR00T predicts.",
    )
    p.add_argument(
        "--expect-dof", type=int, default=rec.get("expected_dof"),
        help="Abort if the first episode's joint dim != this. Default: recording.expected_dof "
             "from scene_config.yaml (None disables the check).",
    )
    return p.parse_args()


def make_dirs(out_dir, camera_name):
    data_dir = os.path.join(out_dir, "data", "chunk-000")
    meta_dir = os.path.join(out_dir, "meta")
    video_dir = os.path.join(
        out_dir, "videos", "chunk-000", f"observation.images.{camera_name}"
    )
    for d in (data_dir, meta_dir, video_dir):
        os.makedirs(d, exist_ok=True)
    return data_dir, meta_dir, video_dir


def write_video(frames, path, fps):
    # Encode with libx264 via the system ffmpeg binary directly. cv2.VideoWriter's
    # "avc1"/"mp4v" fourccs are unreliable here — they either silently fall back to
    # MPEG-4 Part 2 or try a hardware h264_v4l2m2m encoder that fails to open. Most
    # LeRobot/GR00T video loaders expect H.264, so encode that explicitly.
    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg (libx264) failed with exit code {ret} for {path}")


def write_parquet(joint_pos, actions, ep_idx, fps, path):
    T = joint_pos.shape[0]
    df = pd.DataFrame(
        {
            "observation.state": [joint_pos[i].astype(np.float32) for i in range(T)],
            "action": [actions[i].astype(np.float32) for i in range(T)],
            "episode_index": np.full(T, ep_idx, dtype=np.int64),
            "frame_index": np.arange(T, dtype=np.int64),
            "timestamp": (np.arange(T) / float(fps)).astype(np.float32),
            "task_index": np.zeros(T, dtype=np.int64),
        }
    )
    df.to_parquet(path, index=False)
    return T


CHUNKS_SIZE = 1000  # episodes per chunk dir; script writes a single chunk-000 today


def write_meta(meta_dir, args, episode_lengths, state_dim, action_dim, action_horizon):
    total_episodes = len(episode_lengths)
    total_frames = int(sum(episode_lengths))

    info = {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": args.fps,
        "chunks_size": CHUNKS_SIZE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            f"observation.images.{args.camera_name}": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
        },
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep, length in enumerate(episode_lengths):
            f.write(
                json.dumps({"episode_index": ep, "tasks": [0], "length": int(length)})
                + "\n"
            )

    # modality.json — flat top-level schema (video/state/action/annotation) required
    # by GR00T's lerobot_episode_loader. Generated directly from this run's actual
    # camera name and DOF dims rather than copied from a possibly-stale template.
    #
    # Actions stay flat (one row per frame) in the parquet; the prediction horizon is
    # expressed here as action delta_indices = range(action_horizon). GR00T builds the
    # (horizon, action_dim) target at train time from these indices — do NOT bake a
    # lookahead array into the data.
    modality = {
        "video": {
            args.camera_name: {
                "original_key": f"observation.images.{args.camera_name}",
            }
        },
        "state": {
            "joint_positions": {
                "original_key": "observation.state",
                "start": 0,
                "end": state_dim,
                "delta_indices": [0],
            }
        },
        "action": {
            "joint_positions": {
                "original_key": "action",
                "start": 0,
                "end": action_dim,
                "delta_indices": list(range(action_horizon)),
            }
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }
    with open(os.path.join(meta_dir, "modality.json"), "w") as f:
        json.dump(modality, f, indent=2)


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
    if not files:
        print(f"❌ No .h5 files found in {args.input_dir}")
        return

    data_dir, meta_dir, video_dir = make_dirs(args.output_dir, args.camera_name)
    print(f"Found {len(files)} HDF5 episodes → {args.output_dir}")

    episode_lengths = []
    state_dim = action_dim = None
    for ep_idx, fpath in enumerate(files):
        with h5py.File(fpath, "r") as f:
            images = f["observations/images/head_cam"][:]
            joint_pos = f["observations/joint_pos"][:]
            actions = f["actions"][:]

        if state_dim is None:
            state_dim, action_dim = joint_pos.shape[1], actions.shape[1]
            if args.expect_dof is not None and state_dim != args.expect_dof:
                raise SystemExit(
                    f"❌ DOF mismatch: {os.path.basename(fpath)} has joint dim {state_dim}, "
                    f"but --expect-dof is {args.expect_dof}. Wrong --input-dir? "
                    f"(The legacy Inspire dataset is 53 DOF; the current Revo2 dataset is 51.) "
                    f"Pass --expect-dof <N> or set recording.expected_dof to override."
                )
        elif joint_pos.shape[1] != state_dim or actions.shape[1] != action_dim:
            print(
                f"⚠  {os.path.basename(fpath)} dims "
                f"({joint_pos.shape[1]}, {actions.shape[1]}) != first episode's "
                f"({state_dim}, {action_dim}) — check Stage 2 for inconsistent recording"
            )

        mp4_path = os.path.join(video_dir, f"episode_{ep_idx:06d}.mp4")
        write_video(images, mp4_path, args.fps)

        pq_path = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
        T = write_parquet(joint_pos, actions, ep_idx, args.fps, pq_path)

        episode_lengths.append(T)
        print(f"  ✅ {os.path.basename(fpath)} → episode_{ep_idx:06d} ({T} frames)")

    write_meta(meta_dir, args, episode_lengths, state_dim, action_dim, args.action_horizon)
    total_frames = int(sum(episode_lengths))
    print("  ✅ meta files written")
    print(
        f"\n✅ Converted {len(episode_lengths)} episodes, "
        f"{total_frames} total frames → {args.output_dir}"
    )
    print("Ready to transfer to lab server for GR00T fine-tuning.")


if __name__ == "__main__":
    main()
