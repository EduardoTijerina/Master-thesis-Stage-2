#!/usr/bin/env python3
"""Headless render check: load only the G1 wrapper, frame a camera on it, save a PNG."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, Usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api import World

G1 = "/home/eduardot/gr00t_project/assets/g1_inspire_dfq_clean.usd"
OUT = "/home/eduardot/gr00t_project/stage1_scene/_g1_check.png"

world = World()
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=G1, prim_path="/World/G1")
world.reset()

stage = omni.usd.get_context().get_stage()

# world-space bbox of the robot (incl. instance proxies)
bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                       [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                       useExtentsHint=True)
rng = bc.ComputeWorldBound(stage.GetPrimAtPath("/World/G1")).ComputeAlignedRange()
mn, mx = rng.GetMin(), rng.GetMax()
center = (np.array(mn) + np.array(mx)) / 2.0
size = np.array(mx) - np.array(mn)
print(f"BBOX min={tuple(round(v,3) for v in mn)} max={tuple(round(v,3) for v in mx)}")
print(f"CENTER={tuple(round(v,3) for v in center)} SIZE={tuple(round(v,3) for v in size)}")

# count visible meshes via instance proxies
n_mesh = sum(1 for p in Usd.PrimRange(stage.GetPrimAtPath("/World/G1"),
             Usd.TraverseInstanceProxies()) if p.GetTypeName() == "Mesh")
print(f"MESHES_RENDERABLE={n_mesh}")

# place a camera looking at the robot center from a 3 m diagonal
from isaacsim.sensors.camera import Camera
dist = max(size.max() * 1.8, 2.0)
eye = center + np.array([dist, dist, dist * 0.4])
cam = Camera(prim_path="/World/check_cam", resolution=(640, 640))
cam.initialize()
cam.set_world_pose(eye, None)
# aim camera at center
cam_prim = stage.GetPrimAtPath("/World/check_cam")
xf = UsdGeom.Xformable(cam_prim)
# build look-at
import omni.isaac.core.utils.rotations as rot  # noqa
forward = center - eye
forward = forward / np.linalg.norm(forward)
# use SetTransform via Gf matrix look-at
m = Gf.Matrix4d()
m.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0, 0, 1))
m = m.GetInverse()
xf.ClearXformOpOrder()
xf.AddTransformOp().Set(m)

for _ in range(30):
    world.step(render=True)

cam.get_current_frame()
rgba = cam.get_rgba()
print("RGBA shape:", None if rgba is None else rgba.shape)
if rgba is not None and rgba.size:
    try:
        from PIL import Image
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(OUT)
        print(f"SAVED {OUT}")
    except Exception as e:
        print("PIL save failed:", e)
        np.save(OUT.replace(".png", ".npy"), rgba)
        print("saved npy instead")

app.close()
