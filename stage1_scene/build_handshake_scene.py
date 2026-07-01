#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--verify-only", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.usd
from pxr import UsdPhysics, PhysxSchema, Usd, UsdGeom, Gf, Sdf
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.viewports import set_camera_view
import yaml 

with open("/home/eduardot/gr00t_project/configs/scene_config.yaml", "r") as f:
    config = yaml.safe_load(f)


def apply_articulation_root(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"⚠  Prim not found at {prim_path}")
        return
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    PhysxSchema.PhysxArticulationAPI.Apply(prim)
    print(f"✅ ArticulationRootAPI applied to {prim_path}")


def build_scene():
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("get_assets_root_path() returned None.")
    print(f"✅ Assets root: {assets_root}")

    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/10.0)
    world.scene.add_default_ground_plane()
    print("✅ World + ground plane")

    # Load Room from YAML
    room_usd = assets_root + config["scene"]["room_usd"]
    add_reference_to_stage(usd_path=room_usd, prim_path="/World/Room")
    print("✅ Room loaded from config")

    # Use the wrapper (entry point) — it composes base+physics+robot+sensor and exposes
    # geometry as instanceable prims nested under the links. Referencing physics.usd
    # directly drops all meshes (they live in parallel root scopes) and crashes PhysX.
    g1_usd = config["scene"]["g1_usd"] 
    add_reference_to_stage(usd_path=g1_usd, prim_path="/World/G1")
    stage = omni.usd.get_context().get_stage()

    # Standing height: rest-pose foot sole is ~0.792 m below the pelvis origin. The
    # pelvis is rigidly pinned (the fixed joint holds the robot up), so we lift ~1.7 cm
    # above flat-contact to keep the feet just clear of the ground plane. Grazing
    # foot-ground contact under the pinned base + stiff leg drives is over-constrained
    # and makes the solver jitter one leg (asymmetric, non-deterministic).
    STAND_HEIGHT = 0.81
    # NOTE: must set the existing xformOp:translate directly. XformCommonAPI.SetTranslate
    # silently no-ops here because /World/G1 carries an xformOp:orient (quaternion),
    # which XformCommonAPI can't reconcile — the robot would spawn at Z=0 with its feet
    # ~0.79 m underground and PhysX would eject it (the "levitating to 12 m" bug).
    stage.GetPrimAtPath("/World/G1").GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(0.0, 0.0, STAND_HEIGHT)
    )

    # The asset's joint drives ship with stiffness=625 but damping=0 — an undamped
    # spring that pumps energy each physics step and blows the articulation up.
    # Add damping (~critical: 2*sqrt(625)=50) to every angular drive to stabilize it.
    n_damped = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            UsdPhysics.DriveAPI(prim, "angular").GetDampingAttr().Set(50.0)
            n_damped += 1

    # Pin the pelvis to the world with a fixed joint so the robot stands rock-solid
    # (it has no balance controller, so a free base would just collapse to the floor).
    # The physxArticulation:fixBase attribute is ignored on this asset, hence the joint.
    # Anchor the world side at the spawn height so there is no initial constraint violation.
    fj = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/World/G1/base_fixed_joint"))
    fj.CreateBody1Rel().SetTargets([Sdf.Path("/World/G1/pelvis")])
    fj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, STAND_HEIGHT))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fj.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    print(f"✅ G1 loaded: standing @ {STAND_HEIGHT} m, {n_damped} drives damped, base pinned")

    for ext in ("isaacsim.replicator.agent.core", "isaacsim.replicator.agent.ui"):
        try:
            enable_extension(ext)
            print(f"✅ Extension: {ext}")
        except Exception as e:
            print(f"⚠  Could not enable {ext}: {e}")

    world.reset()

    try:
        from isaacsim.sensors.camera import Camera
        
        # Read parameters directly from YAML
        cam_path = config["camera"]["prim_path"]
        cam_res = tuple(config["camera"]["resolution"])

        head_cam = Camera(
            prim_path=cam_path,
            resolution=cam_res,
            # Z=0.48 puts it at eye level relative to the G1 pelvis origin
            # X=0.15 pushes it slightly out of the faceplate so it isn't blind
            
        )
        head_cam.initialize()

        # Inject real-world OAK-D Pro physical sensor optics!
        head_cam.set_focal_length(4.81)           # Standard OAK-D Pro lens
        head_cam.set_horizontal_aperture(6.612)   # IMX378 Sensor width
        head_cam.set_vertical_aperture(5.008)     # IMX378 Sensor height
        
        print(f"✅ head_cam configured to OAK-D Pro physics at {cam_res}")
    except Exception as e:
        print(f"⚠  Camera configuration failed: {e}")

    # Point the viewport camera at the robot so it's framed on startup.
    set_camera_view(eye=[2.4, 2.4, 1.6], target=[0.0, 0.0, 0.9])
    print("✅ Viewport camera framed on G1")

    print("✅ Scene ready")
    return world


def main():
    try:
        world = build_scene()
    except Exception as e:
        print(f"❌ Scene failed: {e}")
        simulation_app.close()
        return

    if args.verify_only:
        for _ in range(60):
            world.step(render=not args.headless)
        print("✅ 60 physics steps — scene verified")
        simulation_app.close()
        return

    print("Scene open. Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    simulation_app.close()


if __name__ == "__main__":
    main()
