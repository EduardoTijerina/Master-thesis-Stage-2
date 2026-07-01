# Dataset Report: EduardoTij/g1-handshake-dataset

**Hub URL:** https://huggingface.co/datasets/EduardoTij/g1-handshake-dataset
**Visibility:** Public
**Format:** LeRobot v2 (parquet + MP4 + JSON metadata)

## Origin

Synthetic robot demonstrations generated in NVIDIA Isaac Sim with a Unitree G1
humanoid (29-DoF body + Inspire dexterous hand) performing a scripted handshake
motion. No real-world data is involved — this is purely simulated, intended as
a sanity-check / pipeline-validation dataset ahead of GR00T VLA fine-tuning.

Pipeline: HDF5 episode capture (Isaac Sim, 10Hz) → converted to LeRobot v2 via
`stage3_export/convert_hdf5_to_lerobot.py`.

## Size and structure

| Metric | Value |
|---|---|
| Total episodes | 100 |
| Total frames | 10,000 (100 frames/episode) |
| FPS | 10 |
| Total size on disk | 63 MB |
| — tabular data (parquet) | 6.3 MB |
| — video (MP4) | 57 MB |
| — metadata (JSON) | 24 KB |
| Total files uploaded | 205 |

```
g1-handshake-dataset/
├── data/chunk-000/episode_000000.parquet ... episode_000099.parquet   (100 files)
├── videos/chunk-000/observation.images.head_cam/*.mp4                  (100 files)
├── meta/info.json
├── meta/episodes.jsonl
├── meta/tasks.jsonl
├── meta/modality.json
└── README.md   (dataset card)
```

## Schema

Each `episode_NNNNNN.parquet` (100 rows = 100 frames):

| Column | Type | Notes |
|---|---|---|
| `observation.state` | `list<float32>`, len 53 | joint positions (body + hand) |
| `action` | `list<float32>`, len 53 | commanded joint positions |
| `episode_index` | `int64` | which episode this row belongs to |
| `frame_index` | `int64` | frame number within episode |
| `timestamp` | `float32` | seconds from episode start |
| `task_index` | `int64` | index into `tasks.jsonl` (always 0 here) |

Video frames are stored separately as MP4 (not embedded in parquet), one
`.mp4` per episode, referenced by `episode_index`/`frame_index`.

| Field | Value |
|---|---|
| `observation.images.head_cam` | video, shape (480, 640, 3), single RGB head camera |

## Task

Single task across all 100 episodes:

```json
{"task_index": 0, "task": "perform a handshake"}
```

## Robot / embodiment

- `robot_type`: `unitree_g1`
- 29-DoF body + Inspire dexterous hand
- State/action dimensionality: 53 (combined body + hand joints)
- Arm motion during collection is a scripted sine-wave placeholder, not a
  learned or IK-based controller — this dataset validates the recording
  pipeline's data shapes/format, not motion quality.

## Known limitations (for downstream consumers / other LLMs reasoning about this data)

- Synthetic/simulated only, no real-robot or human handshake data.
- Single task, single camera view, no domain randomization noted in metadata.
- Motion is a scripted placeholder, not a meaningful demonstration of an actual handshake skill — not yet suitable as training data for a deployable policy.
- No train/val/test split is defined in `meta/` (all 100 episodes are undifferentiated).
