# VLA Fine-Tuning Data42

This repository contains scripts for generating, fine-tuning, and testing the FLOWER Vision-Language-Action (VLA) model.

## Overview

1. **`gen_data42.py`**
   - **Purpose:** Generates the `.npz` episode data.
   - **Contents:** Contains the robot's action records, textual prompts, and camera images (static and gripper views) necessary for training.

2. **`finetune_flower_lora42.py`**
   - **Purpose:** Fine-tunes the base FLOWER VLA model.
   - **Contents:** Reads the episode data generated above, selectively updates specific parameters (like the action generation components), and safely saves the resulting model weights in both `.pt` and `.safetensors` formats.

3. **`test_finetuned_flower42.py`**
   - **Purpose:** Tests and evaluates the fine-tuned FLOWER model in a simulated PyBullet environment.
   - **Contents:** Demonstrates a hybrid AI pipeline. First, the fine-tuned FLOWER model performs a closed-loop approach based on visual inputs and language prompts. Once close to the object, it seamlessly hands over control to a SAC-based multi-finger grasping policy to execute a precise grasp and lift. The entire process is recorded and saved as an MP4 video with overlaid debug information.
