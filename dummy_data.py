import numpy as np
import json
import pandas as pd
from pathlib import Path

def create_dummy_lerobot_dataset(output_dir="dummy_g1_dataset", num_episodes=10, frames_per_ep=20):
    root = Path(output_dir)
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    video_dir = root / "videos" / "chunk-000" / "observation.images.head_cam"

    for d in [meta_dir, data_dir, video_dir]:
        d.mkdir(parents=True, exist_ok=True)

    state_dim = 53   # G1 body (29) + Inspire hands (24) = 53 total DOFs
    action_dim = 53
    fps = 10

    # --- meta/tasks.jsonl ---
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "Perform a handshake with the human."}) + "\n")

    # --- meta/episodes.jsonl ---
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in range(num_episodes):
            f.write(json.dumps({"episode_index": ep, "tasks": [0], "length": frames_per_ep}) + "\n")

    # --- meta/info.json ---
    info = {
        "robot_type": "unitree_g1",
        "fps": fps,
        "total_episodes": num_episodes,
        "total_frames": num_episodes * frames_per_ep,
        "features": {
            "observation.images.head_cam": {"dtype": "video", "shape": [512, 512, 3]},
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
        }
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # --- meta/modality.json (GR00T-specific) ---
    # delta_indices covers all 53 joints — tells GR00T to treat all action dims as deltas
    modality = {
        "video": {
            "head_cam": {"original_key": "observation.images.head_cam"}
        },
        "state": {
            "joint_positions": {
                "original_key": "observation.state",
                "delta_indices": [0]
            }
        },
        "action": {
            "joint_positions": {
                "original_key": "action",
                "delta_indices": list(range(53))
            }
        }
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # --- Per-episode parquet files ---
    for ep in range(num_episodes):
        rows = []
        for frame in range(frames_per_ep):
            rows.append({
                "observation.state": np.random.randn(state_dim).astype(np.float32).tolist(),
                "action": np.random.randn(action_dim).astype(np.float32).tolist(),
                "episode_index": ep,
                "frame_index": frame,
                "timestamp": round(frame / fps, 4),
                "task_index": 0,
                "observation.images.head_cam": f"videos/chunk-000/observation.images.head_cam/episode_{ep:06d}.mp4"
            })
        df = pd.DataFrame(rows)
        df.to_parquet(data_dir / f"episode_{ep:06d}.parquet", index=False)

    print(f"✅ Generated {num_episodes} episodes at ./{output_dir}")
    print(f"   state_dim={state_dim}, action_dim={action_dim}, fps={fps}")
    print(f"   Structure: meta/ + data/chunk-000/ + videos/")

if __name__ == "__main__":
    create_dummy_lerobot_dataset()
