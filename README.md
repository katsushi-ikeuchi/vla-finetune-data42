# VLA Fine-Tuning Data42

This repository contains scripts for generating and fine-tuning the FLOWER Vision-Language-Action (VLA) model.

## Overview

1. **`gen_data42.py`**
   - **Purpose:** Generates the `.npz` episode data.
   - **Contents:** Contains the robot's action records, textual prompts, and camera images (static and gripper views) necessary for training.

2. **`finetune_flower_lora42.py`**
   - **Purpose:** Fine-tunes the base FLOWER VLA model.
   - **Contents:** Reads the episode data generated above, selectively updates specific parameters (like the action generation components), and safely saves the resulting model weights in both `.pt` and `.safetensors` formats.
