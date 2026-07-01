#!/usr/bin/env python3
"""
Stage 0 — Hello World orchestrator.

Sends a single observation to a running GR00T policy server and prints the
returned action chunk + round-trip latency.

Uses GR00T's PolicyClient (ZeroMQ + msgpack under the hood) — do NOT hand-roll
zmq sockets.

Run:
    # syntax check, no server needed:
    python stage0_sanity/orchestrator_hello_world.py --mock
    # real call against a running policy server:
    python stage0_sanity/orchestrator_hello_world.py --server-ip localhost --port 5555
"""

import argparse
import sys
import time

import numpy as np

IMG_H, IMG_W = 512, 512
STATE_DIM = 23
TASK_STRING = "perform a handshake"
LATENCY_THRESHOLD_MS = 150.0


def build_observation():
    """Exact shapes GR00T N1.7 expects for this embodiment."""
    return {
        "video": {
            "head_cam": np.random.randint(
                0, 256, (1, 1, IMG_H, IMG_W, 3), dtype=np.uint8
            )
        },
        "state": {
            "joint_positions": np.zeros((1, 1, STATE_DIM), dtype=np.float32)
        },
        # Language is REQUIRED — the model is a VLA.
        "language": {"task": [[TASK_STRING]]},
    }


def describe(obs):
    print("Observation that WOULD be sent:")
    print(f"  video.head_cam        shape={obs['video']['head_cam'].shape} "
          f"dtype={obs['video']['head_cam'].dtype}")
    print(f"  state.joint_positions shape={obs['state']['joint_positions'].shape} "
          f"dtype={obs['state']['joint_positions'].dtype}")
    print(f"  language.task         {obs['language']['task']}")


def main():
    parser = argparse.ArgumentParser(description="GR00T Hello World orchestrator")
    parser.add_argument("--server-ip", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip the policy server and just print what would be sent.",
    )
    args = parser.parse_args()

    obs = build_observation()

    if args.mock:
        print("🧪 --mock mode: no policy server contacted.\n")
        describe(obs)
        print("\n✅ Mock run complete (syntax OK).")
        return

    try:
        from gr00t.policy.server_client import PolicyClient
    except ImportError as e:
        print(f"❌ Could not import gr00t.policy.server_client: {e}")
        print("   Install with: cd ~/Isaac-GR00T && pip install -e '.[client]'")
        sys.exit(1)

    print(f"Connecting to policy server at {args.server_ip}:{args.port} ...")
    policy = PolicyClient(host=args.server_ip, port=args.port)

    if not policy.ping():
        print("❌ policy.ping() returned False — is the server running?")
        sys.exit(1)
    print("✅ ping OK")

    t0 = time.perf_counter()
    action = policy.get_action(obs)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    # action may be a dict of arrays or a raw array — handle both.
    if isinstance(action, dict):
        arrays = [np.asarray(v) for v in action.values()]
        flat = np.concatenate([a.ravel() for a in arrays])
        shapes = {k: np.asarray(v).shape for k, v in action.items()}
        print(f"Action chunk keys/shapes: {shapes}")
    else:
        arr = np.asarray(action)
        flat = arr.ravel()
        print(f"Action chunk shape: {arr.shape}")

    print(f"First 5 values: {flat[:5]}")
    print(f"Round-trip latency: {latency_ms:.1f} ms")

    if latency_ms <= LATENCY_THRESHOLD_MS:
        print(f"✅ Within {LATENCY_THRESHOLD_MS:.0f}ms threshold")
    else:
        print(f"⚠️  Above {LATENCY_THRESHOLD_MS:.0f}ms threshold")


if __name__ == "__main__":
    main()
