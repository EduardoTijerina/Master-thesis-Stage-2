#!/usr/bin/env python3
"""
Stage 2 — Handshake data collection loop.

Runs the scene from Stage 1 and records episodes to HDF5 at 10Hz. The arm
motion is a scripted sinusoidal placeholder (NOT DiffIK yet) — its only job is
to prove the full image+state+action recording pipeline works end to end.

Run:
    source /home/eduardot/isaac_env/bin/activate
    python stage2_collect/collect_handshake_data.py --num-episodes 100 --headless

HDF5 layout per episode file (episode_NNNN.h5):
    /observations/images/head_cam  (T, 512, 512, 3) uint8
    /observations/joint_pos        (T, STATE_DIM)   float32
    /actions                       (T, ACTION_DIM)  float32
    /timestamps                    (T,)             float64
    /  attr "task" = "perform a handshake"

STATE_DIM/ACTION_DIM are derived at runtime from the loaded robot's DOF
count (set in build_world() after g1.initialize()), not hardcoded — this
USD now carries the Inspire 5-finger hands (23 body + 12 hand DOF).
"""

import argparse
import os

import yaml

CONFIG_PATH = "/home/eduardot/gr00t_project/configs/scene_config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Args + SimulationApp FIRST (before any other omni/isaacsim import).
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Collect G1 handshake episodes")
parser.add_argument("--num-episodes", type=int, default=None,
                    help="Default: recording.max_episodes from scene_config.yaml")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

cfg = load_config()
NUM_EPISODES = args.num_episodes or int(cfg["recording"]["max_episodes"])
FRAMES_PER_EPISODE = int(cfg["recording"]["frames_per_episode"])
FPS = int(cfg["recording"]["fps"])
OUTPUT_DIR = cfg["recording"]["output_dir"]
CAM_RES = tuple(cfg["camera"]["resolution"])
TASK_STRING = cfg["handshake"]["task_string"]
G1_START = cfg["scene"]["g1_start_position"]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": args.headless, "renderer": "RayTracedLighting"}
)

import numpy as np  # noqa: E402
import h5py  # noqa: E402

from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402


def build_world():
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("get_assets_root_path() returned None — check S3 access.")

    world = World(
        physics_dt=float(cfg["scene"]["physics_dt"]),
        rendering_dt=float(cfg["scene"]["render_dt"]),
    )
    world.scene.add_default_ground_plane()
    add_reference_to_stage(
        usd_path=assets_root + cfg["scene"]["room_usd"], prim_path="/World/Room"
    )
    add_reference_to_stage(
        usd_path=cfg["scene"]["g1_usd"], prim_path="/World/G1"
    )

    g1 = Articulation(prim_paths_expr="/World/G1", positions=np.array([G1_START]))

    from isaacsim.sensors.camera import Camera

    head_cam = Camera(
        prim_path=cfg["camera"]["prim_path"],
        translation=np.array([0.0, 0.0, 0.1]),
        resolution=CAM_RES,
    )

    world.reset()
    g1.initialize()

    global STATE_DIM, ACTION_DIM
    STATE_DIM = g1.num_dof
    ACTION_DIM = g1.num_dof

    head_cam.initialize()
    head_cam.set_focal_length(4.81)
    head_cam.set_horizontal_aperture(6.612)
    head_cam.set_vertical_aperture(5.008)

    return world, g1, head_cam


def scripted_action(joint_pos, frame):
    """Hold all joints, wave the right arm (DOF ~12-16) with a sine."""
    action = np.array(joint_pos, dtype=np.float32).copy()
    t = frame / float(FRAMES_PER_EPISODE)
    hi = min(17, action.shape[0])
    for i in range(12, hi):
        action[i] = 0.3 * np.sin(2.0 * np.pi * t)
    return action


def capture_rgb(head_cam):
    rgba = head_cam.get_rgba()
    if rgba is None or np.asarray(rgba).size == 0:
        return np.zeros((CAM_RES[1], CAM_RES[0], 3), dtype=np.uint8)
    return np.asarray(rgba)[..., :3].astype(np.uint8)


def collect_episode(world, g1, head_cam, ep_idx):
    world.reset()
    images, joints, actions, timestamps = [], [], [], []

    for frame in range(FRAMES_PER_EPISODE):
        world.step(render=True)

        img = capture_rgb(head_cam)
        jp = np.asarray(g1.get_joint_positions()).ravel().astype(np.float32)
        if jp.shape[0] < STATE_DIM:  # pad defensively
            jp = np.pad(jp, (0, STATE_DIM - jp.shape[0]))
        jp = jp[:STATE_DIM]

        act = scripted_action(jp, frame)
        try:
            g1.set_joint_position_targets(act)
        except Exception:  # noqa: BLE001
            pass  # placeholder motion is best-effort

        images.append(img)
        joints.append(jp)
        actions.append(act[:ACTION_DIM])
        timestamps.append(frame / float(FPS))

    return (
        np.stack(images).astype(np.uint8),
        np.stack(joints).astype(np.float32),
        np.stack(actions).astype(np.float32),
        np.asarray(timestamps, dtype=np.float64),
    )


def save_episode(ep_idx, images, joints, actions, timestamps):
    path = os.path.join(OUTPUT_DIR, f"episode_{ep_idx:04d}.h5")
    with h5py.File(path, "w") as f:
        obs = f.create_group("observations")
        imgs = obs.create_group("images")
        imgs.create_dataset("head_cam", data=images, compression="gzip")
        obs.create_dataset("joint_pos", data=joints)
        f.create_dataset("actions", data=actions)
        f.create_dataset("timestamps", data=timestamps)
        f.attrs["task"] = TASK_STRING
    return path, images.shape[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Collecting {NUM_EPISODES} episodes × {FRAMES_PER_EPISODE} frames "
          f"@ {FPS}Hz → {OUTPUT_DIR}")

    try:
        world, g1, head_cam = build_world()
        print("✅ Scene ready")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Scene build failed: {e}")
        simulation_app.close()
        return

    try:
        for ep in range(NUM_EPISODES):
            data = collect_episode(world, g1, head_cam, ep)
            path, T = save_episode(ep, *data)
            print(f"✅ Episode {ep} saved: {T} frames, path: {path}")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Collection failed: {e}")
    finally:
        simulation_app.close()
        print("✅ Done.")


if __name__ == "__main__":
    main()
