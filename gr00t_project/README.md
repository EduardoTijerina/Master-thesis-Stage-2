# GR00T N1.7 Handshake Pipeline — Unitree G1 + BrainCo Revo2

Synthetic handshake data generation + fine-tuning pipeline for an NVIDIA GR00T
N1.7-3B policy on a Unitree G1 humanoid fitted with BrainCo Revo2 hands
(51 DOF: 29 body + 22 hand, 11 per hand).

## Machine split

| Role | Machine | Purpose |
|------|---------|---------|
| **Desktop** | Ubuntu 24.04, RTX 5070 Ti (16GB), Isaac Sim 5.1 venv | Scene building, data collection, inference / policy server |
| **Lab server** | Blackwell GPU(s), `Isaac-GR00T` repo | GR00T fine-tuning — **single GPU only**, 2-GPU DeepSpeed ZeRO-2 crashes with CUDA error 700 on Blackwell (no NVLink) |

- Isaac Sim 5.1 venv: `source /home/eduardot/isaac_env/bin/activate`
- Project root: `/home/eduardot/gr00t_project/`
- Isaac Sim API: `from isaacsim import SimulationApp` (NOT `omni.isaac.kit`)
- G1 / room USDs stream from S3 via `get_assets_root_path()` — never hardcoded.
- Robot asset: `assets/g1_revo2_clean.usd` (51 DOF). An older Inspire-hands
  variant (53 DOF) is kept only as `hdf5_episodes_inspire_backup/` — do not
  mix with the Revo2 dataset.

## Layout

```
gr00t_project/
├── stage0_sanity/
│   ├── generate_dummy_dataset.py     # random LeRobot v2 dataset (no Isaac Sim)
│   └── orchestrator_hello_world.py   # single observation → GR00T policy server
├── stage1_scene/
│   ├── build_handshake_scene.py      # Simple Room + G1 + head_cam, interactive builder
│   └── reimport_g1.py                # rebuilds the USD from URDF (after URDF edits)
├── stage2_collect/
│   └── collect_handshake_data.py     # 3-phase scripted trajectory → HDF5 @ 10Hz
├── stage3_export/
│   └── convert_hdf5_to_lerobot.py    # HDF5 → LeRobot v2 parquet + MP4 + meta
├── configs/
│   ├── modality.json                 # reference copy — Stage 3 generates the authoritative one
│   └── scene_config.yaml
├── 3D model/
│   └── g1_with_revo2/g1_with_revo2_hands.urdf   # source URDF (Revo2 mount fix applied)
├── Documentation/
│   ├── PIPELINE_ARCHITECTURE.md
│   ├── HF_DATASET_REPORT.md
│   └── GR00T_Pipeline_Architecture_Report.pdf
└── README.md
```

## One-time setup

```bash
source /home/eduardot/isaac_env/bin/activate
pip install pandas pyarrow h5py opencv-python-headless pyyaml

# GR00T client (orchestrator only):
pip install zmq msgpack
git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T
cd ~/Isaac-GR00T && pip install -e ".[client]"
```

> Use `opencv-python-headless` (no display server in headless mode).
> System `python3` lacks `h5py` — always launch Stage 2/3 with
> `~/isaac_env/bin/python`, not a bare `python`.

## Run order

### Stage 0 — Sanity check (run BEFORE touching Isaac Sim)

```bash
source /home/eduardot/isaac_env/bin/activate
python stage0_sanity/generate_dummy_dataset.py
# → Transfer dummy_g1_dataset/ to lab server, run a 5-step fine-tune there.
# → Transfer the checkpoint back, launch the policy server, then:
python stage0_sanity/orchestrator_hello_world.py --mock   # test syntax
python stage0_sanity/orchestrator_hello_world.py          # real test
```

### Stage 1 — Build scene

```bash
python stage1_scene/build_handshake_scene.py --verify-only
```

The G1 **is stabilized**, not left to balance itself: pelvis raised to
`STAND_HEIGHT = 0.81` (feet clear of the ground), all drive joints damped
(asset ships kp=625/kd=0, undamped otherwise), and the pelvis pinned in
place with a `UsdPhysics.FixedJoint` — there is no balance controller, so an
unpinned base tips over from arm reaction torque. Stage 2 must replicate all
three steps exactly (`stabilize_g1()`), or the robot falls over during
collection. Camera modeled on OAK-D Pro optics (focal_length=4.81,
h_aperture=6.612, v_aperture=5.008).

### Stage 2 — Collect data

```bash
~/isaac_env/bin/python stage2_collect/collect_handshake_data.py --num-episodes 1000 --headless --seed 42
```

Each 100-frame episode (10s @ 10Hz) runs a **3-phase scripted trajectory**
(no DiffIK): a fixed rest pose, a cosine-eased reach into the handshake arm
pose, and a hand-grip close — not a sine-wave placeholder. Useful preview
flags (GUI only, not for production runs):

| Flag | Effect |
|------|--------|
| `--play-trajectory` | loop the full 3-phase motion live, FPS-paced |
| `--show-camera` | open a window showing exactly what gets recorded to HDF5 |
| `--hold-pose` | freeze at the final (grip) pose |
| `--hold-rest` | freeze at the starting rest pose |
| `--tune-left` | declared, not yet implemented |

### Stage 3 — Export to LeRobot v2

```bash
~/isaac_env/bin/python stage3_export/convert_hdf5_to_lerobot.py --expect-dof 51
```

Defaults: reads `/mnt/data/gr00t_data/hdf5_episodes`, writes
`/mnt/data/gr00t_data/handshake_dataset`. `--expect-dof 51` aborts the run if
pointed at the wrong (53-DOF Inspire) directory — pass explicit
`--input-dir`/`--output-dir` to override.

### Dataset status

The current Revo2 51-DOF dataset — **1000 episodes / 100,000 frames**,
converted to LeRobot v2 — is ready at `/mnt/data/gr00t_data/handshake_dataset/`.
HuggingFace upload is pending (username not yet confirmed):

```bash
huggingface-cli login   # once
huggingface-cli upload <hf-username>/handshake_dataset \
  /mnt/data/gr00t_data/handshake_dataset/ --repo-type dataset
```

> An older 23-DOF/sine-wave-placeholder dataset was previously published at
> `EduardoTij/g1-handshake-dataset` — that run predates the Revo2 hands and
> the 3-phase trajectory and is **superseded** by the dataset above.

### Transfer to lab server

```bash
huggingface-cli download <hf-username>/handshake_dataset \
  --repo-type dataset --local-dir ~/datasets/handshake_dataset/
# Fallback if not uploaded yet:
rsync -avz /mnt/data/gr00t_data/handshake_dataset/ \
  USER@LAB_SERVER_IP:~/datasets/handshake_dataset/
```

## Lab server — fine-tuning (run ON THE SERVER)

```bash
cd ~/Isaac-GR00T
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path ~/Isaac-GR00T/checkpoints/GR00T-N1.7-3B \
  --dataset-path ~/datasets/handshake_dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ~/Isaac-GR00T/stage0_sanity/new_embodiment_config.py \
  --num-gpus 1 \
  --output-dir ~/checkpoints/gr00t-handshake \
  --max-steps 50000 --save-steps 5000 --global-batch-size 16
```

- `NEW_EMBODIMENT` is a post-train tag that only exists once this fine-tune
  produces it — `UNITREE_G1` does not exist yet.
- Base checkpoint must be the local `GR00T-N1.7-3B` dir — N1-2B is incompatible.
- **Single GPU only** (see Machine split above); `NCCL_P2P_DISABLE=1` does not
  fix the 2-GPU crash.

## Desktop — policy server (after syncing checkpoint back)

```bash
rsync -avz user@lab:~/checkpoints/gr00t-handshake/checkpoint-50000/ \
  ~/checkpoints/gr00t-handshake/checkpoint-50000/
source /home/eduardot/isaac_env/bin/activate
cd ~/Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path ~/checkpoints/gr00t-handshake/checkpoint-50000 \
  --embodiment-tag NEW_EMBODIMENT \
  --port 5556
```

> Use port **5556**, not 5555 — 5555 is frequently taken by other users on
> the shared lab server.

## Data formats

**HDF5 (Stage 2 output)** — `episode_NNNN.h5`:
```
/observations/images/head_cam   (T, 512, 512, 3) uint8
/observations/joint_pos         (T, 51)          float32
/actions                        (T, 51)          float32
/timestamps                     (T,)             float64
/  attr "task" = "perform a handshake"
```

**LeRobot v2 (Stage 3 output)** — parquet columns: `observation.state`,
`action`, `episode_index`, `frame_index`, `timestamp`, `task_index`; videos as
MP4 (mp4v @ 10fps); meta = `info.json`, `episodes.jsonl`, `tasks.jsonl`,
`modality.json` (state `delta_indices=[0]`, action `delta_indices=range(40)`
matching GR00T N1.7's AH=40).
