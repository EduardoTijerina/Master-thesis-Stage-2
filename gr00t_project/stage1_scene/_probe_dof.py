#!/usr/bin/env python3
"""Probe the actual DOF count and joint names of g1_revo2_clean.usd."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation

USD_PATH = "/home/eduardot/gr00t_project/assets/g1_revo2_clean.usd"

world = World()
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/G1")
world.reset()

robot = Articulation(prim_paths_expr="/World/G1")
world.step(render=False)   # physics must tick before initialize()
robot.initialize()

with open("/tmp/dof_probe_out.txt", "w") as fh:
    fh.write(f"num_dof: {robot.num_dof}\n")
    for i, name in enumerate(robot.dof_names):
        fh.write(f"{i:2d} {name}\n")

app.close()
