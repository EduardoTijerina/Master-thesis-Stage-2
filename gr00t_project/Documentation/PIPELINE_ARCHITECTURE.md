# Pipeline Internals — How Stages 1–3 Work and Talk to Each Other

This document explains *how* each stage script works internally and *how*
data and config flow between them. For setup/run commands, see `README.md`.
For known bugs found during the 2026-06-19 audit, see the bottom section —
they are referenced inline where relevant.

## 1. The shared contract: `configs/scene_config.yaml`

Every stage re-reads this file independently — there is no shared Python
module, so each script's reach into the YAML is a hand-maintained string
path that has to match the others by convention, not by code.

```yaml
scene:     room_usd, g1_usd, g1_start_position, physics_dt, render_dt
camera:    name, prim_path, resolution
recording: fps, output_dir, max_episodes, frames_per_episode
handshake: human_approach_distance, contact_duration_frames, task_string
```

- **Stage 1** (`build_handshake_scene.py`) reads `scene.room_usd`, `scene.g1_usd`,
  `camera.prim_path`, `camera.resolution`.
- **Stage 2** (`collect_handshake_data.py`) reads almost the whole file:
  `recording.*`, `camera.resolution`, `handshake.task_string`,
  `scene.g1_start_position`, `scene.room_usd`, `scene.g1_usd`,
  `scene.physics_dt`, `scene.render_dt`.
- **Stage 3** (`convert_hdf5_to_lerobot.py`) reads `recording.{fps,action_horizon,
  expected_dof}` from `scene_config.yaml` as CLI-overridable defaults, and infers
  `state_dim`/`action_dim` from the first episode's actual arrays. It **generates**
  `meta/modality.json` itself (it does not copy `configs/modality.json`); that
  generated file is the authoritative one GR00T's `lerobot_episode_loader` consumes.
  `configs/modality.json` is a kept-in-sync reference of the same flat schema.

⚠️ Stage 1 and Stage 2 resolve `scene.g1_usd` differently — Stage 1 uses it
as a literal path, Stage 2 prefixes it with `assets_root`. See **Known
issues** below; this means the two stages do not currently load the same
robot asset.

## 2. Stage 1 — `build_handshake_scene.py` (interactive scene builder)

Purpose: stand up the Isaac Sim scene once, visually, so you can confirm the
room/robot/camera look right before running unattended collection.

Execution order (matters because of Isaac Sim's import constraints):

1. Parse `--headless` / `--verify-only` **before** importing anything Isaac
   related.
2. Construct `SimulationApp` — this must happen before any `omni.*` /
   `isaacsim.*` import, otherwise those modules don't exist yet.
3. `build_scene()`:
   - Create a `World` with `physics_dt=1/60`, `rendering_dt=1/10`.
   - Add the default ground plane.
   - Reference the room USD under `/World/Room` (path = `assets_root +
     scene.room_usd`).
   - Reference the G1 USD under `/World/G1` (path = `scene.g1_usd` as-is).
   - Translate `/World/G1` up to `STAND_HEIGHT = 0.793` m so the feet sit on
     the ground plane (must touch `xformOp:translate` directly — see code
     comment on why `XformCommonAPI` silently no-ops here).
   - Walk every prim with an angular `DriveAPI` and set damping to 50 — the
     asset ships with stiffness=625/damping=0, which is an undamped spring
     that explodes the articulation within a few physics steps without this.
   - Add a `FixedJoint` pinning the pelvis to the world, since there's no
     balance controller and an unpinned base just collapses.
   - Enable two Replicator extensions (best-effort, exceptions are caught
     and logged, not fatal).
   - `world.reset()` to apply all of the above.
   - Create the `Camera` prim at `camera.prim_path` with `camera.resolution`,
     then explicitly set OAK-D Pro real-world optics (`focal_length=4.81mm`,
     `horizontal_aperture=6.612mm`, `vertical_aperture=5.008mm`) so the
     simulated camera matches the physical sensor that will eventually
     replace it on hardware.
   - Frame the *viewport* camera (not the head_cam) on the robot for visual
     inspection.
4. `main()`: if `--verify-only`, step physics 60 times headless/rendered and
   exit (this is the CI-style smoke test). Otherwise loop `world.step()`
   forever until Ctrl+C — this is the "leave it open and look at it" mode.

Stage 1 produces nothing on disk. It either exits 0 (scene is sane) or
crashes with a printed `❌`. Its only job is risk reduction before Stage 2
runs unattended for ~100 episodes.

## 3. Stage 2 — `collect_handshake_data.py` (unattended recording loop)

Purpose: run the same scene non-interactively and record episodes to HDF5.

Execution order:

1. Parse `--num-episodes` / `--headless`, then immediately `load_config()`
   and pull every constant it needs (`NUM_EPISODES`, `FRAMES_PER_EPISODE`,
   `FPS`, `OUTPUT_DIR`, `CAM_RES`, `TASK_STRING`, `G1_START`) into module
   globals **before** importing `isaacsim` — same ordering constraint as
   Stage 1.
2. `STATE_DIM = 23` / `ACTION_DIM = 23` are hardcoded here, not derived from
   the loaded robot (see Known issues — this silently caps how many joints
   get recorded).
3. `build_world()`: re-does the same three `add_reference_to_stage` calls as
   Stage 1 (room + G1) but **without** any of Stage 1's damping fix-up,
   fixed-joint pinning, or camera optics calls — it relies on the USD itself
   already being physically stable (i.e., it assumes Stage 1 already proved
   the scene works, it doesn't redo Stage 1's fixes). It wraps the G1 in an
   `Articulation` object instead of just a raw prim reference, because Stage
   2 needs to read/write joint state every frame, which `Articulation` gives
   direct vectorized access to.
4. `collect_episode()` — the per-episode hot loop, run `NUM_EPISODES` times:
   - `world.reset()` at the start of every episode (so each episode starts
     from the same pose, since there's no real task variation yet).
   - For each of `FRAMES_PER_EPISODE` frames:
     - `world.step(render=True)` — advance physics + render one frame.
     - `capture_rgb()` — pull the camera's RGBA buffer, drop alpha, cast to
       `uint8`; returns a black frame if the camera isn't ready yet (avoids
       crashing on the first frame or two before the render pipeline warms
       up).
     - Read joint positions from the `Articulation`, pad/truncate to
       `STATE_DIM`.
     - `scripted_action()` — the placeholder motion: copies current joint
       positions, then overwrites DOF indices 12–16 with a sine wave scaled
       to the episode's normalized time `t = frame/FRAMES_PER_EPISODE`. This
       is **not** a real handshake — it's a fixed sinusoid standing in for
       wherever the real DiffIK/motion-retargeting controller will go later,
       so the rest of the pipeline (recording, HDF5 schema, LeRobot export)
       can be built and tested independently of that controller.
     - `g1.set_joint_position_targets(act)` — actually drives the robot;
       wrapped in a bare `try/except` because this is explicitly
       best-effort placeholder motion, not the final controller.
     - Append image/joint/action/timestamp to in-memory lists.
   - Returns four stacked numpy arrays for the whole episode.
5. `save_episode()` — writes one HDF5 file per episode:
   `episode_NNNN.h5` with groups `/observations/images/head_cam`,
   `/observations/joint_pos`, top-level `/actions`, `/timestamps`, and a
   `task` attribute on the file root.
6. `main()` creates `OUTPUT_DIR` if missing, builds the world once, then
   loops `collect_episode` → `save_episode` for every episode, printing a
   per-episode `✅`/`❌` line. The world/robot/camera objects are reused
   across all 100 episodes — only `world.reset()` runs per-episode, the
   Isaac Sim app itself starts once and stays up for the whole run.

Output: `hdf5_episodes/episode_0000.h5` … `episode_0099.h5` (filename count
driven by `--num-episodes` / `recording.max_episodes`).

## 4. Stage 3 — `convert_hdf5_to_lerobot.py` (offline format converter)

Purpose: turn the directory of per-episode HDF5 files into the LeRobot v2
layout GR00T's fine-tuning script expects. This stage has **no Isaac Sim
dependency** — it's pure `h5py`/`cv2`/`pandas`, which is why it can run
anywhere, including on the lab server directly if you skip the rsync step.

Execution order:

1. `parse_args()` — all paths/dims/fps are CLI flags with defaults matching
   `scene_config.yaml`'s values, but **not actually read from the YAML** —
   if you change `state_dim` in the YAML you must also pass `--state-dim` here
   or the defaults silently drift out of sync.
2. `make_dirs()` — creates the LeRobot v2 skeleton:
   `data/chunk-000/`, `meta/`, `videos/chunk-000/observation.images.<camera_name>/`.
3. For each `episode_*.h5` file found (sorted, so episode index order is
   filename order, not creation order):
   - Open the HDF5, pull `images`, `joint_pos`, `actions` arrays into memory.
   - `write_video()` — re-encodes the image stack to MP4 (`mp4v` codec) at
     the target fps. Note the explicit `RGB2BGR` conversion: HDF5 stores RGB
     (as captured from the simulated camera) but OpenCV's `VideoWriter`
     expects BGR, so skipping this conversion would produce a blue/red
     channel-swapped video that looks wrong on playback but would not error.
   - `write_parquet()` — builds the per-episode dataframe with columns
     `observation.state`, `action`, `episode_index`, `frame_index`,
     `timestamp`, `task_index`, and writes it straight to parquet.
   - Track each episode's frame count for the meta files written after the
     loop.
4. `write_meta()` — after all episodes are converted:
   - `info.json` — dataset-level stats (`total_episodes`, `total_frames`,
     `fps`) plus a `features` block describing the shape/dtype of each
     observation/action column. Image shape is `[480, 640, 3]` and
     state/action shapes come from the inferred DOF count.
   - `tasks.jsonl` — single line mapping `task_index 0` → the task string.
   - `episodes.jsonl` — one line per episode: index, task list, length.
   - `modality.json` — **generated here** (flat `video/state/action/annotation`
     schema) from this run's actual camera name and DOF dims. Actions stay flat
     in the parquet; the GR00T prediction horizon is expressed as action
     `delta_indices = range(action_horizon)` (default 40, from
     `recording.action_horizon`). This generated file — not
     `configs/modality.json` — is what GR00T loads; `configs/modality.json` is a
     synced reference of the same schema.

Output: `handshake_dataset/{data,meta,videos}/...`, ready to `rsync` to the
lab server.

## 5. End-to-end data flow

```
scene_config.yaml ─┬─► build_handshake_scene.py  (interactive check, writes nothing)
                    └─► collect_handshake_data.py
                          │
                          ├─ reads room_usd/g1_usd (own path-resolution logic, see §1)
                          ├─ reads camera.prim_path/resolution
                          ├─ STATE_DIM/ACTION_DIM derived from g1.num_dof (51 for Revo2)
                          │
                          ▼
                   hdf5_episodes/episode_NNNN.h5  (×100, one per episode)
                          │
                          ▼
              convert_hdf5_to_lerobot.py
                          │
              ┌───────────┼─────────────────┐
              ▼           ▼                 ▼
       videos/*.mp4  data/*.parquet   meta/{info,tasks,episodes,modality}.json
                          │
                          ▼
              rsync → lab server → GR00T fine-tuning (README.md, "Lab server" section)
```

The only thing that crosses a process boundary between stages is the
filesystem: Stage 2 writes HDF5 files, Stage 3 reads them back from disk.
There's no in-memory handoff and no shared Python code between stages —
each script is a fully separate `python <script>.py` invocation, which is
why config drift between them (see below) isn't caught until runtime.

## 6. Known issues (carried over from the 2026-06-19 audit)

These affect the interactions described above and are worth fixing before
the next real collection run:

1. ~~`scene.g1_usd` path-resolution mismatch (Stage 1 literal vs Stage 2
   `assets_root`-prefixed)~~ **RESOLVED**: both stages now load `scene.g1_usd`
   as a literal path, pointing at the custom `assets/g1_revo2_clean.usd`
   (51 DOF). Only `scene.room_usd` is `assets_root`-prefixed.
2. ~~`STATE_DIM`/`ACTION_DIM = 23` hardcoded~~ **RESOLVED**: Stage 2 derives
   both from `g1.num_dof` after `initialize()`; Stage 3 infers from the first
   episode's arrays and now guards with `--expect-dof` (default 51).
3. ~~Resolution mismatch (512×512 vs 640×480)~~ **RESOLVED**: `scene_config.yaml`,
   the recorded HDF5, Stage 3's `info.json` features, and `modality.json` all
   agree on 640×480 (`[480,640,3]` HxWxC). Stale "512×512" mentions in docstrings
   have been corrected.
4. **Stage 2's camera never gets the OAK-D Pro optics** that Stage 1 sets
   (`set_focal_length`/`set_horizontal_aperture`/`set_vertical_aperture`) —
   Stage 2's `Camera(...)` call only sets `translation` and `resolution`.
