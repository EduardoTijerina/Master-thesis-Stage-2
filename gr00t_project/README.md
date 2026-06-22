# GR00T N1.7 Handshake Pipeline — Unitree G1

Synthetic handshake data generation + fine-tuning pipeline for an NVIDIA GR00T
N1.7-3B policy on the Unitree G1 humanoid.

## Machine split

| Role | Machine | Purpose |
|------|---------|---------|
| **Desktop** | Ubuntu 24.04, RTX 5070 Ti (16GB), Isaac Sim 5.1 venv | Scene building, data collection, inference / policy server |
| **Lab server** | 2× GPU, `groot_env` | GR00T fine-tuning |

- Isaac Sim 5.1 venv: `source /home/eduardot/isaac_env/bin/activate`
- Project root: `/home/eduardot/gr00t_project/`
- Isaac Sim API: `from isaacsim import SimulationApp` (NOT `omni.isaac.kit`)
- G1 / room USDs stream from S3 via `get_assets_root_path()` — never hardcoded.

## Layout

```
gr00t_project/
├── stage0_sanity/
│   ├── generate_dummy_dataset.py     # random LeRobot v2 dataset (no Isaac Sim)
│   └── orchestrator_hello_world.py   # single observation → GR00T policy server
├── stage1_scene/
│   └── build_handshake_scene.py      # Simple Room + G1 + head_cam
├── stage2_collect/
│   └── collect_handshake_data.py     # 10Hz recording loop → HDF5
├── stage3_export/
│   └── convert_hdf5_to_lerobot.py    # HDF5 → LeRobot v2 parquet + MP4 + meta
├── configs/
│   ├── modality.json                 # GR00T column → modality mapping
│   └── scene_config.yaml
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

The G1 has **no balance controller** and will fall over — expected. We record
joint *commands* (actions), not stable poses.

### Stage 2 — Collect data

```bash
python stage2_collect/collect_handshake_data.py --num-episodes 100 --headless
```

Arm motion is a scripted sine-wave placeholder (not DiffIK yet) — it exists only
to prove the image+state+action recording pipeline works.

### Stage 3 — Export to LeRobot v2

```bash
python stage3_export/convert_hdf5_to_lerobot.py \
  --input-dir ./hdf5_episodes \
  --output-dir ./handshake_dataset
```

### Transfer to lab server

```bash
rsync -avz --progress ./handshake_dataset/ \
  USER@LAB_SERVER_IP:/home/USER/datasets/handshake_dataset/
```

## Lab server — fine-tuning (run ON THE SERVER)

```bash
source ~/groot_env/bin/activate
cd ~/Isaac-GR00T
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/datasets/handshake_dataset \
  --embodiment-tag new_embodiment \
  --modality-config-path ~/datasets/handshake_dataset/meta/modality.json \
  --num-gpus 2 \
  --output-dir ~/checkpoints/gr00t-handshake \
  --max-steps 5 \
  --global-batch-size 4 \
  --tune-llm False \
  --tune-visual False \
  --tune-projector True \
  --tune-diffusion-model True
# --max-steps 5 for sanity; raise to 2000+ for real training.
```

## Desktop — policy server (after transferring checkpoint back)

```bash
source /home/eduardot/isaac_env/bin/activate
cd ~/Isaac-GR00T
python gr00t/eval/run_gr00t_server.py \
  --model-path ~/checkpoints/gr00t-handshake \
  --embodiment-tag new_embodiment \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

## Data formats

**HDF5 (Stage 2 output)** — `episode_NNNN.h5`:
```
/observations/images/head_cam   (T, 512, 512, 3) uint8
/observations/joint_pos         (T, 23)          float32
/actions                        (T, 23)          float32
/timestamps                     (T,)             float64
/  attr "task" = "perform a handshake"
```

**LeRobot v2 (Stage 3 output)** — parquet columns: `observation.state`,
`action`, `episode_index`, `frame_index`, `timestamp`, `task_index`; videos as
MP4 (mp4v @ 10fps); meta = `info.json`, `episodes.jsonl`, `tasks.jsonl`,
`modality.json`.
