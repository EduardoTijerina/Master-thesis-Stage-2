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
Stage 2 recorded (body + Inspire hand joints) without needing to know it
in advance.
"""

import argparse
import glob
import json
import os

import cv2
import h5py
import numpy as np
import pandas as pd

CONFIG_MODALITY = "/home/eduardot/gr00t_project/configs/modality.json"


def parse_args():
    p = argparse.ArgumentParser(description="Convert HDF5 episodes to LeRobot v2")
    p.add_argument("--input-dir", default="/home/eduardot/gr00t_project/hdf5_episodes")
    p.add_argument("--output-dir", default="/home/eduardot/gr00t_project/handshake_dataset")
    p.add_argument("--task", default="perform a handshake")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--camera-name", default="head_cam")
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
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for {path}")
    for frame in frames:
        # HDF5 stores RGB; cv2 expects BGR for correct colours on playback.
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


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


def write_meta(meta_dir, args, episode_lengths, state_dim, action_dim):
    total_episodes = len(episode_lengths)
    total_frames = int(sum(episode_lengths))

    info = {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": args.fps,
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

    # modality.json — copy from configs/ if present.
    if os.path.exists(CONFIG_MODALITY):
        with open(CONFIG_MODALITY) as f:
            modality = json.load(f)
    else:
        modality = {
            "observation": {
                "images": {
                    args.camera_name: {
                        "original_key": f"observation.images.{args.camera_name}",
                        "delta_indices": [0],
                        "shape": [3, 512, 512],
                    }
                },
                "state": {
                    "joint_positions": {
                        "original_key": "observation.state",
                        "delta_indices": [0],
                        "shape": [state_dim],
                        "dtype": "float32",
                    }
                },
            },
            "action": {
                "joint_positions": {
                    "original_key": "action",
                    "delta_indices": list(range(15)),
                    "shape": [action_dim],
                    "dtype": "float32",
                }
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

    write_meta(meta_dir, args, episode_lengths, state_dim, action_dim)
    total_frames = int(sum(episode_lengths))
    print("  ✅ meta files written")
    print(
        f"\n✅ Converted {len(episode_lengths)} episodes, "
        f"{total_frames} total frames → {args.output_dir}"
    )
    print("Ready to transfer to lab server for GR00T fine-tuning.")


if __name__ == "__main__":
    main()
