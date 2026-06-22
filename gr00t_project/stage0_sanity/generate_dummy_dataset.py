#!/usr/bin/env python3
"""
Stage 0 — Dummy LeRobot v2 dataset generator.

Creates a complete LeRobot v2 dataset filled with RANDOM data (no Isaac Sim
required). Use this to validate the GR00T fine-tuning script on the lab server
*before* spending GPU hours generating real handshake data.

Output: /home/eduardot/gr00t_project/dummy_g1_dataset/

Run:
    source /home/eduardot/isaac_env/bin/activate
    python stage0_sanity/generate_dummy_dataset.py
"""

import json
import os

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed parameters (match the GR00T / Unitree G1 spec in the prompt)
# ---------------------------------------------------------------------------
OUTPUT_DIR = "/home/eduardot/gr00t_project/dummy_g1_dataset"
NUM_EPISODES = 10
FRAMES_PER_EPISODE = 20
FPS = 10
STATE_DIM = 23
ACTION_DIM = 23
IMG_H, IMG_W = 512, 512
CAMERA_NAME = "head_cam"
TASK_STRING = "perform a handshake"

CONFIG_MODALITY = "/home/eduardot/gr00t_project/configs/modality.json"


def make_dirs():
    data_dir = os.path.join(OUTPUT_DIR, "data", "chunk-000")
    meta_dir = os.path.join(OUTPUT_DIR, "meta")
    video_dir = os.path.join(
        OUTPUT_DIR, "videos", "chunk-000",
        f"observation.images.{CAMERA_NAME}",
    )
    for d in (data_dir, meta_dir, video_dir):
        os.makedirs(d, exist_ok=True)
    return data_dir, meta_dir, video_dir


def write_parquet(data_dir, ep_idx):
    """One row per frame, columns per the LeRobot v2 spec."""
    rows = {
        "observation.state": [
            np.random.randn(STATE_DIM).astype(np.float32)
            for _ in range(FRAMES_PER_EPISODE)
        ],
        "action": [
            np.random.randn(ACTION_DIM).astype(np.float32)
            for _ in range(FRAMES_PER_EPISODE)
        ],
        "episode_index": np.full(FRAMES_PER_EPISODE, ep_idx, dtype=np.int64),
        "frame_index": np.arange(FRAMES_PER_EPISODE, dtype=np.int64),
        "timestamp": (np.arange(FRAMES_PER_EPISODE) / float(FPS)).astype(np.float32),
        "task_index": np.zeros(FRAMES_PER_EPISODE, dtype=np.int64),
    }
    df = pd.DataFrame(rows)
    path = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
    df.to_parquet(path, index=False)
    return path


def write_video(video_dir, ep_idx):
    """Random uint8 frames encoded to MP4 with the mp4v codec."""
    path = os.path.join(video_dir, f"episode_{ep_idx:06d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(FPS), (IMG_W, IMG_H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for {path}")
    for _ in range(FRAMES_PER_EPISODE):
        frame = np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def write_meta(meta_dir):
    total_frames = NUM_EPISODES * FRAMES_PER_EPISODE

    info = {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1",
        "total_episodes": NUM_EPISODES,
        "total_frames": total_frames,
        "fps": FPS,
        "features": {
            f"observation.images.{CAMERA_NAME}": {
                "dtype": "video",
                "shape": [IMG_H, IMG_W, 3],
            },
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
        },
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_STRING}) + "\n")

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in range(NUM_EPISODES):
            f.write(
                json.dumps(
                    {"episode_index": ep, "tasks": [0], "length": FRAMES_PER_EPISODE}
                )
                + "\n"
            )

    # modality.json — copy from configs if present, else write the canonical one.
    modality = {
        "observation": {
            "images": {
                CAMERA_NAME: {
                    "original_key": f"observation.images.{CAMERA_NAME}",
                    "delta_indices": [0],
                    "shape": [3, IMG_H, IMG_W],
                }
            },
            "state": {
                "joint_positions": {
                    "original_key": "observation.state",
                    "delta_indices": [0],
                    "shape": [STATE_DIM],
                    "dtype": "float32",
                }
            },
        },
        "action": {
            "joint_positions": {
                "original_key": "action",
                "delta_indices": list(range(15)),
                "shape": [ACTION_DIM],
                "dtype": "float32",
            }
        },
    }
    if os.path.exists(CONFIG_MODALITY):
        with open(CONFIG_MODALITY) as f:
            modality = json.load(f)
    with open(os.path.join(meta_dir, "modality.json"), "w") as f:
        json.dump(modality, f, indent=2)


def verify_checklist(data_dir, meta_dir, video_dir):
    print("\n=== Verification checklist ===")
    ok = True

    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
        p = os.path.join(meta_dir, name)
        exists = os.path.exists(p)
        ok &= exists
        print(f"  {'✅' if exists else '❌'} meta/{name}")

    for ep in range(NUM_EPISODES):
        pq = os.path.join(data_dir, f"episode_{ep:06d}.parquet")
        mp = os.path.join(video_dir, f"episode_{ep:06d}.mp4")
        pq_ok, mp_ok = os.path.exists(pq), os.path.exists(mp)
        ok &= pq_ok and mp_ok
        print(
            f"  {'✅' if pq_ok else '❌'} data/chunk-000/episode_{ep:06d}.parquet"
            f"   {'✅' if mp_ok else '❌'} video"
        )

    print("\n" + ("✅ Dummy dataset complete." if ok else "❌ Missing files — see above."))
    return ok


def main():
    print(f"Generating dummy LeRobot v2 dataset → {OUTPUT_DIR}")
    data_dir, meta_dir, video_dir = make_dirs()

    for ep in range(NUM_EPISODES):
        write_parquet(data_dir, ep)
        write_video(video_dir, ep)
        print(f"  ✅ episode {ep:06d}: {FRAMES_PER_EPISODE} frames")

    write_meta(meta_dir)
    print("  ✅ meta files written")

    verify_checklist(data_dir, meta_dir, video_dir)


if __name__ == "__main__":
    main()
