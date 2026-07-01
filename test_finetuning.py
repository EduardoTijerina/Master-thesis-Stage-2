import numpy as np
import time
from gr00t.policy.server_client import PolicyClient

def run_orchestrator_sanity_check(server_ip="localhost", port=5555):
    print(f"Connecting to GR00T Policy Server at {server_ip}:{port}...")
    policy = PolicyClient(host=server_ip, port=port, timeout_ms=15000)

    if not policy.ping():
        raise RuntimeError("❌ Cannot reach policy server. Is it running on port 5555?")
    print("✅ Policy server reachable.")

    observation = {
        "video": {
            "head_cam": np.random.randint(0, 255, (1, 1, 512, 512, 3), dtype=np.uint8)
        },
        "state": {
            "joint_positions": np.zeros((1, 1, 53), dtype=np.float32)
        },
        "language": {
            "instruction": [["perform a handshake"]]
        }
    }

    print("Sending observation to GR00T policy server...")
    start = time.perf_counter()
    action, info = policy.get_action(observation)
    latency_ms = (time.perf_counter() - start) * 1000

    print("\n--- Sanity Check Results ---")
    print(f"Action type:        {type(action)}")
    print(f"Action chunk shape: {action.shape}")  # expect (1, horizon, 53)
    print(f"Round-trip latency: {latency_ms:.2f} ms")
    print(f"Info keys:          {list(info.keys()) if isinstance(info, dict) else info}")

    if latency_ms < 150:
        print("✅ SUCCESS: Within 150ms real-time budget.")
    else:
        print(f"⚠️  WARNING: {latency_ms:.0f}ms exceeds budget. "
              "Consider TensorRT/FP8 quantization.")

if __name__ == "__main__":
    run_orchestrator_sanity_check()
