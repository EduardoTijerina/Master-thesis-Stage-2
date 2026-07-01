#!/usr/bin/env python3
"""
Stage 2 — Handshake data collection (scripted reach-to-target, v2).

3-phase joint-space trajectory per episode:
  Phase 1 (frames  0-19, 20%):  Hold rest pose.
  Phase 2 (frames 20-79, 60%):  Cosine-eased smooth reach to handshake arm pose.
  Phase 3 (frames 80-99, 20%):  Progressive right-hand grip close (arm holds target).

Episode randomization:
  ±JITTER_RAD (0.05 rad) applied independently to each of the 7 right-arm
  target angles before interpolation.  At ~50 cm arm length this gives
  ~±2.5 cm Cartesian spread at the wrist — sufficient for GR00T to learn
  causal position tracking without requiring live IK.

Right-hand drive gains: stiffness=40, damping=0.8 (compliant grip).

All joint indices (right arm: 7, right hand: 11) are derived at runtime from
the articulation's joint-name list — nothing is hardcoded.

HDF5 layout per episode file (episode_NNNN.h5):
    /observations/images/head_cam  (T, 480, 640, 3)  uint8
    /observations/joint_pos        (T, 51)            float32
    /actions                       (T, 51)            float32
    /timestamps                    (T,)               float64
    /  attr "task" = "perform a handshake"

Run:
    source /home/eduardot/isaac_env/bin/activate
    cd /home/eduardot/gr00t_project
    # validation pass:
    python stage2_collect/collect_handshake_data.py --num-episodes 5 --headless
    # production run:
    python stage2_collect/collect_handshake_data.py --num-episodes 1000 --headless
"""

import argparse
import os
import time

import numpy as np
import yaml

CONFIG_PATH = "/home/eduardot/gr00t_project/configs/scene_config.yaml"

# ── Handshake arm target pose (joint-space, radians) ──────────────────────────
# Sign conventions for G1 Revo2 (verified against URDF / probe):
#   shoulder_pitch: negative = forward elevation
#   shoulder_roll:  negative = adduction (arm toward body centreline)
#   elbow:          positive = flexion
#   wrist_*:        near-zero for a neutral handshake angle
# Cartesian intent: right hand extends to ~[0.5, -0.3, 0.9] m (local frame).
# Tune after a visual inspection run (--headless=False, 5 episodes) if needed.
HANDSHAKE_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.45,  # moderate forward flexion (not raised high)
    "right_shoulder_roll_joint":  -0.10,  # slight adduction toward centreline
    "right_shoulder_yaw_joint":    0.10,  # slight inward rotation
    "right_elbow_joint":           0.6,  # ~49° flexion → forearm raised higher (was 1.10)
    "right_wrist_roll_joint":      1.57,  # roll about forearm axis (±1.97) → palm faces LEFT,
                                          #   not forward (a real handshake hand, thumb up)
    "right_wrist_pitch_joint":     0.00,
    "right_wrist_yaw_joint":       0.00,
}

# ── Left-arm rest pose: fully extended down against the thigh ──────────────────
# The left arm stays here for the whole episode (it plays no part in the handshake).
# Fingers are left at 0 (extended/straight) — no left-hand entries needed.
REST_LEFT_ARM_POSE = {
    "left_shoulder_pitch_joint":  0.00,
    "left_shoulder_roll_joint":   0.30,
    "left_shoulder_yaw_joint":    0.00,
    "left_elbow_joint":          1.3,  # lower URDF limit → most extended (straight)
    "left_wrist_roll_joint":      -1.5,    # reset for now; tune palm direction after arm is straight
    "left_wrist_pitch_joint":     0.00,
    "left_wrist_yaw_joint":       0.00,
}

# Starting position of the right arm (Phase 1 rest, before the reach begins).
# All zeros = arm hangs straight down. Tune these if you want the arm to begin
# in a different neutral pose (e.g. slightly bent elbow, rotated wrist).
REST_RIGHT_ARM_POSE = {
    "right_shoulder_pitch_joint": 0.00,
    "right_shoulder_roll_joint":  0.00,
    "right_shoulder_yaw_joint":   0.00,
    "right_elbow_joint":          1.00,
    "right_wrist_roll_joint":     1.00,
    "right_wrist_pitch_joint":    0.00,
    "right_wrist_yaw_joint":      0.00,
}

# ── Right-hand grip close targets (radians) ───────────────────────────────────
# Targets chosen WITHIN each joint's URDF limit and coherent with the Revo2 mimic
# ratios (distal ≈ 1.155 × proximal; thumb_distal ≈ thumb_proximal) so the curl
# looks like a closing fist instead of a broken claw.
#   proximal fingers limit 1.41 | distal limit 1.63 | thumb_* limits 1.03–1.57
HANDSHAKE_HAND_POSE = {
    "right_index_proximal_joint":   0.20,   # distal mimic follows at ×1.155 → ~1.56 rad
    "right_middle_proximal_joint":  0.35,
    "right_ring_proximal_joint":    0.35,
    "right_pinky_proximal_joint":   0.35,
    "right_index_distal_joint":     0.50,   # mimic-driven; this target is effectively ignored
    "right_middle_distal_joint":    0.50,   # keeping it here for action-space completeness
    "right_ring_distal_joint":      0.50,
    "right_pinky_distal_joint":     0.50,
    "right_thumb_metacarpal_joint": 0.60,
    "right_thumb_proximal_joint":   1.0,
    # right_thumb_distal_joint intentionally omitted: it's a mimic joint with a
    # 1.1 N·m effort cap that collision forces always overwhelm, driving it negative.
    # Leaving it undriven keeps it at its rest position (0 rad).
}

# ── Trajectory / randomisation parameters ────────────────────────────────────
JITTER_RAD    = 0.05   # ±rad per arm joint, ≈ ±2.5 cm at wrist
PHASE1_END    = 0.20   # fraction of episode: hold rest
PHASE2_END    = 0.80   # fraction of episode: reach (then grip to 1.0)

# Drive gains: the asset ships stiffness=625 / damping=0 (undamped → explodes).
# We add damping=50 (≈critical for kp=625) to EVERY angular drive at the USD level
# in stabilize_g1(), matching Stage 1. Fingers stay firm (kp=625) so the grip
# actually closes — the earlier soft kp=40 let the 2 N·m-effort-capped distal
# (mimic) joints get shoved the wrong way. Compliant hand gains can be re-introduced
# later, AFTER the motion is visually confirmed correct.
DRIVE_DAMPING = 50.0


# ── Config / argparse (before any omni import) ────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


cfg = load_config()
rec = cfg["recording"]

parser = argparse.ArgumentParser(description="Collect G1 handshake episodes (scripted reach)")
parser.add_argument("--num-episodes", type=int, default=int(rec["max_episodes"]))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seed", type=int, default=42, help="RNG seed for episode jitter reproducibility")
parser.add_argument("--hold-pose", action="store_true",
                    help="Skip collection: apply the FINAL handshake pose (right arm reached + "
                         "hand closed + left arm down) and hold it forever for visual tuning. "
                         "Orbit the GUI camera to inspect; Ctrl+C to exit.")
parser.add_argument("--tune-left", action="store_true",
                    help="Hold RIGHT arm at handshake pose but leave left arm FREE so you can "
                         "drag it manually via the Articulation Inspector. Prints left-arm "
                         "joint values every 3 seconds so you can copy them into the config.")
parser.add_argument("--play-trajectory", action="store_true",
                    help="Skip collection: play the full 3-phase handshake trajectory "
                         "(rest → reach with open hand → grip close) on a loop in the GUI, "
                         "using the nominal (no-jitter) pose. For visual review before a run.")
parser.add_argument("--hold-rest", action="store_true",
                    help="Skip collection: hold the STARTING (rest) pose forever — right arm "
                         "at 0 (hanging down), left arm down at its side. Orbit the GUI camera "
                         "to inspect the right arm at frame 0 for jitter/glitching. Ctrl+C to exit.")
parser.add_argument("--show-camera", action="store_true",
                    help="Use with --play-trajectory: open a second window showing the robot's "
                         "head-camera POV in real time as the trajectory plays.")
args = parser.parse_args()

np.random.seed(args.seed)

NUM_EPISODES       = args.num_episodes
FRAMES_PER_EPISODE = int(rec["frames_per_episode"])
FPS                = int(rec["fps"])
OUTPUT_DIR         = rec["output_dir"]
CAM_RES            = tuple(cfg["camera"]["resolution"])   # (W, H) = (640, 480)
TASK_STRING        = cfg["handshake"]["task_string"]
G1_START           = cfg["scene"]["g1_start_position"]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import h5py  # noqa: E402
import omni.usd  # noqa: E402

from pxr import UsdPhysics, Sdf, Gf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

# Standing height: rest-pose foot sole is ~0.792 m below the pelvis origin. The
# pelvis is rigidly pinned (the fixed joint is what holds the robot up), so we lift
# ~1.7 cm above flat-contact to keep the feet just clear of the ground plane.
# Grazing foot-ground contact under the pinned base + stiff leg drives is
# over-constrained and makes the solver jitter one leg (asymmetric, non-deterministic).
STAND_HEIGHT = 0.81


# ── Runtime globals (set in build_world after robot.initialize()) ─────────────
STATE_DIM          = None
ACTION_DIM         = None
RIGHT_ARM_INDICES  = None   # list[int], len 7
RIGHT_HAND_INDICES = None   # list[int], len 11
LEFT_ARM_INDICES   = None   # list[int], len 7
REST_POSE_51       = None   # np.ndarray(STATE_DIM): base pose held every frame
                            #   (zeros everywhere except the left arm down at its side)


# ── DOF utilities ─────────────────────────────────────────────────────────────

def get_indices_by_name(dof_names, wanted):
    """Return DOF indices for each name in *wanted*, in that order."""
    name_to_idx = {n: i for i, n in enumerate(dof_names)}
    missing = [n for n in wanted if n not in name_to_idx]
    if missing:
        raise RuntimeError(f"Joints not found in articulation: {missing}")
    return [name_to_idx[n] for n in wanted]


def stabilize_g1(stage):
    """
    Replicate Stage 1's base stabilization (must run BEFORE world.reset()):
      1. Lift the pelvis to STAND_HEIGHT so the feet rest on the ground plane.
      2. Add damping to every angular drive (asset ships kp=625, kd=0 → explodes).
      3. Pin the pelvis to the world with a fixed joint (no balance controller,
         so a free base tips over from the arm's reaction torque).
    Returns the number of drives damped.
    """
    # 1. Spawn height — set the existing xformOp:translate directly (XformCommonAPI
    #    no-ops because /World/G1 carries an xformOp:orient quaternion).
    stage.GetPrimAtPath("/World/G1").GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(0.0, 0.0, STAND_HEIGHT)
    )

    # 2. Damping on every angular drive.
    n_damped = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            UsdPhysics.DriveAPI(prim, "angular").GetDampingAttr().Set(DRIVE_DAMPING)
            n_damped += 1

    # 3. Pin the pelvis to the world at the spawn height (no initial constraint
    #    violation), making this a fixed-base manipulator for handshake demos.
    fj = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/World/G1/base_fixed_joint"))
    fj.CreateBody1Rel().SetTargets([Sdf.Path("/World/G1/pelvis")])
    fj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, STAND_HEIGHT))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fj.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    return n_damped


# ── Scene construction ────────────────────────────────────────────────────────

def build_world():
    global STATE_DIM, ACTION_DIM, RIGHT_ARM_INDICES, RIGHT_HAND_INDICES
    global LEFT_ARM_INDICES, REST_RIGHT_ARM_INDICES, REST_POSE_51

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("get_assets_root_path() returned None — check S3 access.")

    world = World(
        physics_dt=float(cfg["scene"]["physics_dt"]),
        rendering_dt=float(cfg["scene"]["render_dt"]),
    )
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=assets_root + cfg["scene"]["room_usd"], prim_path="/World/Room")
    add_reference_to_stage(usd_path=cfg["scene"]["g1_usd"], prim_path="/World/G1")

    # Base stabilization on the USD stage — MUST happen before world.reset().
    stage = omni.usd.get_context().get_stage()
    n_damped = stabilize_g1(stage)
    print(f"✅ Stabilized: standing @ {STAND_HEIGHT} m, {n_damped} drives damped, pelvis pinned")

    g1 = Articulation(prim_paths_expr="/World/G1")

    head_cam = Camera(
        prim_path=cfg["camera"]["prim_path"],
        translation=np.array([0.0, 0.0, 0.1]),
        resolution=CAM_RES,
    )

    world.reset()
    g1.initialize()

    STATE_DIM = ACTION_DIM = g1.num_dof
    dof_names = list(g1.dof_names)

    RIGHT_ARM_INDICES      = get_indices_by_name(dof_names, list(HANDSHAKE_ARM_POSE.keys()))
    RIGHT_HAND_INDICES     = get_indices_by_name(dof_names, list(HANDSHAKE_HAND_POSE.keys()))
    LEFT_ARM_INDICES       = get_indices_by_name(dof_names, list(REST_LEFT_ARM_POSE.keys()))
    REST_RIGHT_ARM_INDICES = get_indices_by_name(dof_names, list(REST_RIGHT_ARM_POSE.keys()))

    # Base pose held every frame: zeros (legs/torso rest) + both arms at their rest positions.
    REST_POSE_51 = np.zeros(STATE_DIM, dtype=np.float32)
    REST_POSE_51[LEFT_ARM_INDICES]       = np.array(list(REST_LEFT_ARM_POSE.values()),  dtype=np.float32)
    REST_POSE_51[REST_RIGHT_ARM_INDICES] = np.array(list(REST_RIGHT_ARM_POSE.values()), dtype=np.float32)

    # Make world.reset() restore the REST pose (left arm down) instead of the all-zero
    # default — so the left arm is at REST_LEFT_ARM_POSE from frame 0 and never swings in.
    g1.set_joints_default_state(positions=REST_POSE_51.reshape(1, -1))

    print(f"✅ DOF: {STATE_DIM}")
    print(f"   right-arm  indices: {RIGHT_ARM_INDICES}")
    print(f"   right-hand indices: {RIGHT_HAND_INDICES}")
    print(f"   left-arm   indices: {LEFT_ARM_INDICES}")

    head_cam.initialize()
    head_cam.set_focal_length(4.81)
    head_cam.set_horizontal_aperture(6.612)
    head_cam.set_vertical_aperture(5.008)

    return world, g1, head_cam


# ── Trajectory generation ─────────────────────────────────────────────────────

def build_episode_target():
    """
    51-DOF target array for one episode.
    Right-arm joints get HANDSHAKE_ARM_POSE + per-episode jitter.
    Right-hand joints get HANDSHAKE_HAND_POSE (no jitter — grip is consistent).
    All other joints remain at 0 (rest).
    """
    target = np.zeros(STATE_DIM, dtype=np.float32)
    arm_base = np.array(list(HANDSHAKE_ARM_POSE.values()), dtype=np.float32)
    jitter   = np.random.uniform(-JITTER_RAD, JITTER_RAD, len(arm_base)).astype(np.float32)
    target[RIGHT_ARM_INDICES]  = arm_base + jitter
    target[RIGHT_HAND_INDICES] = np.array(list(HANDSHAKE_HAND_POSE.values()), dtype=np.float32)
    return target, jitter


def _cosine_ease(t: float) -> float:
    """Smooth cosine ease-in/out factor: t ∈ [0,1] → [0,1]."""
    return 0.5 * (1.0 - np.cos(np.pi * t))


def trajectory_action(frame: int, episode_target: np.ndarray) -> np.ndarray:
    """
    Return the 51-DOF action for *frame* given *episode_target*.

    Phase 1 — hold rest  (frames 0 .. p1-1)
    Phase 2 — arm reach  (frames p1 .. p2-1)  cosine-eased
    Phase 3 — grip close (frames p2 .. T-1)   cosine-eased; arm holds target
    """
    p1 = int(FRAMES_PER_EPISODE * PHASE1_END)
    p2 = int(FRAMES_PER_EPISODE * PHASE2_END)

    # Base pose every frame: legs/torso rest + left arm down at its side.
    action = REST_POSE_51.copy()

    if frame < p1:
        pass  # Phase 1: hold rest (left arm already down via REST_POSE_51)

    elif frame < p2:
        # Phase 2: smooth arm reach. Interpolate FROM the rest pose TO the target so
        # the reach starts exactly where Phase 1 left the arm (no snap-to-zero jerk).
        t     = (frame - p1) / float(p2 - p1)
        alpha = _cosine_ease(t)
        action[RIGHT_ARM_INDICES] = (
            (1.0 - alpha) * REST_POSE_51[RIGHT_ARM_INDICES]
            + alpha * episode_target[RIGHT_ARM_INDICES]
        )

    else:
        # Phase 3: arm holds; hand closes
        action[RIGHT_ARM_INDICES]  = episode_target[RIGHT_ARM_INDICES]
        t     = (frame - p2) / float(FRAMES_PER_EPISODE - p2)
        alpha = _cosine_ease(t)
        action[RIGHT_HAND_INDICES] = alpha * episode_target[RIGHT_HAND_INDICES]

    return action


# ── Data capture / saving ─────────────────────────────────────────────────────

def capture_rgb(head_cam):
    rgba = head_cam.get_rgba()
    if rgba is None or np.asarray(rgba).size == 0:
        return np.zeros((CAM_RES[1], CAM_RES[0], 3), dtype=np.uint8)
    return np.asarray(rgba)[..., :3].astype(np.uint8)


def reset_to_rest(world, g1):
    """Reset the sim, then TELEPORT the joints to the rest pose.

    world.reset() snaps the articulation to its all-zero default, which leaves the
    left arm straight; the drive would then swing it into REST_LEFT_ARM_POSE during
    Phase 1. Setting positions (not just targets) makes every episode/loop START with
    the left arm already down at its side and the right arm at 0 — so the left arm
    never moves and Phase 1 is a true static rest.
    """
    world.reset()
    # world.reset() invalidates the physics view until the next step, so a teleport
    # issued right now would silently no-op (set_joint_positions needs a live view) and
    # the arms would stay at the all-zero default — that's the "starts extended" flash.
    # Step once WITHOUT rendering to re-establish the view (the zero pose is never shown),
    # then teleport to the rest pose. set_joint_positions needs shape (M,K) = (1, num_dof).
    world.step(render=False)
    g1.set_joint_positions(REST_POSE_51.reshape(1, -1))
    g1.set_joint_position_targets(REST_POSE_51.reshape(1, -1))


def collect_episode(world, g1, head_cam, episode_target):
    reset_to_rest(world, g1)
    images, joints, actions, timestamps = [], [], [], []

    for frame in range(FRAMES_PER_EPISODE):
        world.step(render=True)

        img = capture_rgb(head_cam)
        jp  = np.asarray(g1.get_joint_positions()).ravel().astype(np.float32)
        if jp.shape[0] < STATE_DIM:
            jp = np.pad(jp, (0, STATE_DIM - jp.shape[0]))
        jp = jp[:STATE_DIM]

        act = trajectory_action(frame, episode_target)
        try:
            g1.set_joint_position_targets(act)
        except Exception:  # noqa: BLE001
            pass

        images.append(img)
        joints.append(jp)
        actions.append(act)
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
        obs  = f.create_group("observations")
        imgs = obs.create_group("images")
        imgs.create_dataset("head_cam", data=images, compression="gzip")
        obs.create_dataset("joint_pos", data=joints)
        f.create_dataset("actions",    data=actions)
        f.create_dataset("timestamps", data=timestamps)
        f.attrs["task"] = TASK_STRING
    return path, images.shape[0]


# ── Schema validation helper ──────────────────────────────────────────────────

def validate_episode_file(path):
    """Checks HDF5 schema for a single episode; returns True if clean."""
    try:
        with h5py.File(path, "r") as f:
            jp  = f["observations/joint_pos"][:]
            act = f["actions"][:]
            img = f["observations/images/head_cam"][:]
            assert jp.shape[1]  == 51,    f"joint_pos dim {jp.shape[1]} != 51"
            assert act.shape[1] == 51,    f"actions dim {act.shape[1]} != 51"
            assert img.shape[1] == CAM_RES[1], f"img H {img.shape[1]} != {CAM_RES[1]}"
            assert img.shape[2] == CAM_RES[0], f"img W {img.shape[2]} != {CAM_RES[0]}"
        return True
    except Exception as e:
        print(f"   ❌ validation failed for {path}: {e}")
        return False


# ── Pose-hold inspection mode ─────────────────────────────────────────────────

def hold_pose(world, g1):
    """Apply the final handshake pose and hold it indefinitely for visual tuning."""
    final = REST_POSE_51.copy()
    final[RIGHT_ARM_INDICES]  = np.array(list(HANDSHAKE_ARM_POSE.values()), dtype=np.float32)
    final[RIGHT_HAND_INDICES] = np.array(list(HANDSHAKE_HAND_POSE.values()), dtype=np.float32)

    reset_to_rest(world, g1)
    print("\n🔍 Holding FINAL handshake pose. Orbit the viewport to inspect.")
    print("   Right arm: reached + palm-left + grip closed | Left arm: down at side")
    print("   Ctrl+C (or close the window) to exit.\n")
    try:
        while simulation_app.is_running():
            g1.set_joint_position_targets(final)
            world.step(render=True)
    except KeyboardInterrupt:
        pass


def hold_rest(world, g1):
    """Hold the STARTING (rest) pose indefinitely so the right arm at frame 0 can be inspected.

    This is exactly the pose every episode/loop starts from: right arm at 0 (hanging
    straight down), left arm down at its side via REST_LEFT_ARM_POSE. If the right arm
    looks like it's glitching at the start of the trajectory, freeze it here and orbit
    the viewport — a jitter that persists in this static hold is a stabilization issue
    (drive gains / grazing contact), not a trajectory problem.
    """
    reset_to_rest(world, g1)
    print("\n🔍 Holding STARTING rest pose. Orbit the viewport to inspect the right arm.")
    print("   Right arm: at 0 (hanging down) | Left arm: down at side")
    print("   Ctrl+C (or close the window) to exit.\n")
    try:
        while simulation_app.is_running():
            g1.set_joint_position_targets(REST_POSE_51.reshape(1, -1))
            world.step(render=True)
    except KeyboardInterrupt:
        pass


def play_trajectory(world, g1, head_cam=None):
    """Loop the full 3-phase handshake trajectory in the GUI for visual review.

    Uses the nominal (zero-jitter) target so what you see is the canonical motion
    every episode reaches. Same trajectory_action() the real collection uses, so
    this is a faithful preview — not a separate code path.

    If head_cam is provided (via --show-camera), a second OpenCV window shows the
    robot's head-camera POV in real time.
    """
    import cv2

    # Nominal target: handshake pose with NO jitter (the centre of every episode).
    target = np.zeros(STATE_DIM, dtype=np.float32)
    target[RIGHT_ARM_INDICES]  = np.array(list(HANDSHAKE_ARM_POSE.values()), dtype=np.float32)
    target[RIGHT_HAND_INDICES] = np.array(list(HANDSHAKE_HAND_POSE.values()), dtype=np.float32)

    p1 = int(FRAMES_PER_EPISODE * PHASE1_END)
    p2 = int(FRAMES_PER_EPISODE * PHASE2_END)
    HOLD_FRAMES = 25  # linger on the closed-grip end pose so the loop reads clearly
    frame_dt = 1.0 / float(FPS)  # wall-clock pace so the motion is visible (~10 Hz → 10 s)

    def step_paced(act, frame_label=""):
        """One rendered step, paced to real time so the playback isn't a blur."""
        t0 = time.time()
        g1.set_joint_position_targets(act)
        world.step(render=True)
        if head_cam is not None:
            img = capture_rgb(head_cam)
            # OpenCV expects BGR; add a frame label so it's clear what phase we're in.
            disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if frame_label:
                cv2.putText(disp, frame_label, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Head Camera POV", disp)
            cv2.waitKey(1)
        dt = frame_dt - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)

    print("\n▶  Playing handshake trajectory on a loop. Orbit to the third-person view.")
    if head_cam is not None:
        print("   📷 Head-camera window open ('Head Camera POV').")
    print(f"   Phase 1 rest (0–{p1-1}) → Phase 2 reach, hand OPEN ({p1}–{p2-1}) → "
          f"Phase 3 grip close ({p2}–{FRAMES_PER_EPISODE-1})  | ~{FRAMES_PER_EPISODE/FPS:.0f}s/loop")
    print("   Ctrl+C (or close the window) to exit.\n")
    try:
        loop = 0
        while simulation_app.is_running():
            reset_to_rest(world, g1)
            for frame in range(FRAMES_PER_EPISODE):
                if frame < p1:
                    label = f"Phase 1: rest  [f{frame}]"
                elif frame < p2:
                    label = f"Phase 2: reach [f{frame}]"
                else:
                    label = f"Phase 3: grip  [f{frame}]"
                step_paced(trajectory_action(frame, target), label)
                if not simulation_app.is_running():
                    break
            # Hold the final closed-grip handshake briefly before replaying.
            for _ in range(HOLD_FRAMES):
                if not simulation_app.is_running():
                    break
                step_paced(trajectory_action(FRAMES_PER_EPISODE - 1, target), "Phase 3: grip [HOLD]")
            loop += 1
            print(f"   ↻ replay #{loop}")
    except KeyboardInterrupt:
        pass
    finally:
        if head_cam is not None:
            cv2.destroyAllWindows()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Collecting {NUM_EPISODES} episodes × {FRAMES_PER_EPISODE} frames "
          f"@ {FPS}Hz  →  {OUTPUT_DIR}  (seed={args.seed})")

    try:
        world, g1, head_cam = build_world()
        print("✅ Scene ready")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Scene build failed: {e}")
        simulation_app.close()
        return

    if args.hold_pose:
        hold_pose(world, g1)
        simulation_app.close()
        return

    if args.hold_rest:
        hold_rest(world, g1)
        simulation_app.close()
        return

    if args.play_trajectory:
        play_trajectory(world, g1, head_cam=head_cam if args.show_camera else None)
        simulation_app.close()
        return

    failed = 0
    try:
        for ep in range(NUM_EPISODES):
            episode_target, jitter = build_episode_target()
            data = collect_episode(world, g1, head_cam, episode_target)
            path, T = save_episode(ep, *data)
            ok = validate_episode_file(path)
            status = "✅" if ok else "⚠ "
            print(f"{status} Episode {ep:04d} | {T} frames | "
                  f"arm jitter: {np.round(jitter, 3).tolist()}")
            if not ok:
                failed += 1
    except Exception as e:  # noqa: BLE001
        print(f"❌ Collection loop failed at episode {ep}: {e}")
        import traceback; traceback.print_exc()
    finally:
        simulation_app.close()
        print(f"\n✅ Done. {NUM_EPISODES - failed}/{NUM_EPISODES} episodes validated clean.")


if __name__ == "__main__":
    main()
