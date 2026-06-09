import numpy as np
import os
from PIL import Image
from datasets import Dataset, Features, Sequence, Value, Image as HFImage

def create_dummy_lerobot_dataset(output_dir="dummy_g1_dataset", num_episodes=10, frames_per_ep=20):
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        "observation.image": [],
        "observation.state": [],
        "action": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": []
    }
    
    # Unitree G1 approximate dimensions
    state_dim = 23  # Joint positions
    action_dim = 23 # Joint targets
    
    for ep in range(num_episodes):
        for frame in range(frames_per_ep):
            # Generate fake 512x512 RGB image
            fake_img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            fake_img = Image.fromarray(fake_img_array)
            
            data["observation.image"].append(fake_img)
            data["observation.state"].append(np.random.randn(state_dim).tolist())
            data["action"].append(np.random.randn(action_dim).tolist())
            data["episode_index"].append(ep)
            data["frame_index"].append(frame)
            data["timestamp"].append(frame * 0.1) # 10Hz dummy timestamp

    # Define strict features for LeRobot compatibility
    features = Features({
        "observation.image": HFImage(),
        "observation.state": Sequence(Value("float32")),
        "action": Sequence(Value("float32")),
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
        "timestamp": Value("float32")
    })
    
    dataset = Dataset.from_dict(data, features=features)
    dataset.save_to_disk(output_dir)
    print(f"Successfully generated {num_episodes} dummy episodes at ./{output_dir}")

if __name__ == "__main__":
    create_dummy_lerobot_dataset()