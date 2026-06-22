#!/usr/bin/env python3
"""Re-import the G1 Inspire DFQ URDF into a clean, self-contained USD."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd
from isaacsim.asset.importer.urdf import _urdf

URDF = "/home/eduardot/unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
OUT_USD = "/home/eduardot/gr00t_project/assets/g1_inspire_dfq_clean.usd"

cfg = _urdf.ImportConfig()
# print available attributes for debugging
attrs = [a for a in dir(cfg) if not a.startswith("_")]
print("ImportConfig attrs:", attrs)

cfg.merge_fixed_joints = True
cfg.convex_decomp = False
cfg.import_inertia_tensor = True
cfg.fix_base = False
cfg.self_collision = False
cfg.create_physics_scene = False   # no embedded PhysicsScene → no conflict with World
cfg.make_default_prim = True

print(f"Importing {URDF} ...")
result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF,
    import_config=cfg,
    dest_path=OUT_USD,
)
if result:
    print(f"✅ Imported to {OUT_USD}  (prim: {prim_path})")
else:
    print("❌ Import failed")

app.close()
