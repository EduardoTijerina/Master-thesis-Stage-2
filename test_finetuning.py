import zmq
import time
import numpy as np
import json
import cv2

def run_orchestrator_sanity_check(server_ip="localhost", port="5555"):
    # 1. Setup ZeroMQ Context
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_ip}:{port}")
    print(f"Connected to GR00T Policy Server at {server_ip}:{port}")

    # 2. Prepare Static Dummy Inputs
    # Simulating a 512x512 RGB camera feed and 23-DOF joint state
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_state = np.zeros(23).tolist()

    # If your pipeline expects base64 or JPEG encoded images over the network (recommended for latency):
    _, buffer = cv2.imencode('.jpg', dummy_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    encoded_image = buffer.tolist()

    request_payload = {
        "observation": {
            "image": encoded_image,
            "state": dummy_state
        }
    }

    # 3. Execute Latency Benchmark
    print("Sending dummy observation to VLA...")
    start_time = time.perf_counter()
    
    socket.send_json(request_payload)
    
    # Wait for the action chunk reply
    reply = socket.recv_json()
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    # 4. Evaluate against 150ms budget
    action_chunk = reply.get("action", [])
    print("\n--- Sanity Check Results ---")
    print(f"Action Chunk Received: {len(action_chunk)} dimensions")
    print(f"Round-Trip Latency: {latency_ms:.2f} ms")
    
    if latency_ms < 150:
        print("✅ SUCCESS: Latency is within the 150 ms real-time safety budget.")
    else:
        print("⚠️ WARNING: Latency exceeded 150 ms. Edge optimization (TensorRT/FP8) will be required.")

if __name__ == "__main__":
    run_orchestrator_sanity_check()