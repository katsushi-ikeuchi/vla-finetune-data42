#!/usr/bin/env python3
"""
FLOWER VLA Fine-Tuning Script with Robust Weight Saving
======================================================
15エポックのファインチューニングを実施し、
共有メモリテンソルを正しくハンドリングして保存します。
"""

import os
import sys
import glob
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from omegaconf import OmegaConf
from safetensors.torch import save_model, save_file, load_file
from torch.utils.tensorboard import SummaryWriter

WORKSPACE_DIR = "/home/ikeuchi/robotis"
FLOWER_DIR = "/home/ikeuchi/flower/flower_vla_calvin"
DATA_DIR = os.path.join(WORKSPACE_DIR, "vla/data42/episodes")
CHECKPOINT_SAVE_DIR = os.path.join(WORKSPACE_DIR, "vla/data42/checkpoints/flower_hx5_finetuned_rescaled")

if FLOWER_DIR not in sys.path:
    sys.path.append(FLOWER_DIR)

from flower.models.flower import FLOWERVLA


class MultiFingerEpisodeDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, chunk_size=10):
        self.files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        self.chunk_size = chunk_size
        self.samples = []

        print("[Dataset] Indexing files for lazy loading...")
        for ep_idx, f in enumerate(self.files):
            data = np.load(f, mmap_mode="r")
            T = len(data["actions"])
            
            for t in range(T):
                self.samples.append({
                    "file_path": f,
                    "t": t,
                    "T": T
                })
        print(f"[Dataset] Found {len(self.samples)} total samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        f = item["file_path"]
        t = item["t"]
        T = item["T"]
        
        data = np.load(f)
        
        static_img = data["rgb_static"][t]
        wrist_img = data["rgb_gripper"][t]
        actions = data["actions"]
        lang_text = str(data["lang_text"] if "lang_text" in data else data["prompt"])
        
        if t + self.chunk_size <= T:
            act_chunk = actions[t : t + self.chunk_size]
        else:
            act_chunk = np.zeros((self.chunk_size, 7), dtype=np.float32)
            avail = actions[t:]
            act_chunk[:len(avail)] = avail
            act_chunk[len(avail):] = avail[-1]
            
        t_static = torch.from_numpy(static_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t_wrist = torch.from_numpy(wrist_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t_actions = torch.from_numpy(act_chunk).float()

        return {
            "rgb_static": t_static,
            "rgb_gripper": t_wrist,
            "actions": t_actions,
            "lang_text": lang_text
        }


def custom_collate_fn(batch):
    rgb_static = torch.cat([b["rgb_static"] for b in batch], dim=0).unsqueeze(1)
    rgb_gripper = torch.cat([b["rgb_gripper"] for b in batch], dim=0).unsqueeze(1)
    actions = torch.stack([b["actions"] for b in batch], dim=0)
    lang_texts = [b["lang_text"] for b in batch]

    return {
        "rgb_obs": {
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper
        },
        "actions": actions,
        "lang_text": lang_texts
    }


def train_flower_adaptation(
    num_epochs: int = 15,
    batch_size: int = 8,
    lr: float = 2.0e-5,
    device: str = "cuda:0"
):
    print("\n" + "="*65)
    print("  Starting FLOWER VLA Fine-Tuning (Multi-Finger Adaptation)")
    print("="*65)

    os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(CHECKPOINT_SAVE_DIR, "logs"))
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    dataset = MultiFingerEpisodeDataset(data_dir=DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    config_path = os.path.join(FLOWER_DIR, "pretrained/flower_calvin_d/config.yaml")
    # RESUME FROM EPOCH 7
    model_path = "/home/ikeuchi/flower/flower_vla_calvin/pretrained/flower_calvin_d/model.safetensors"

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model
    if "_target_" in model_cfg:
        del model_cfg["_target_"]
    if "_recursive_" in model_cfg:
        del model_cfg["_recursive_"]

    model_cfg["pretrained_model_path"] = None
    model_cfg["load_pretrained"] = False
    model_cfg["query_seq_len"] = 120

    print("\n[Model] Instantiating FLOWERVLA model...")
    model = FLOWERVLA(**model_cfg)

    print(f"[Model] Loading base weights from: {model_path}")
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    trainable_params = []
    for name, param in model.named_parameters():
        if "dit" in name or "action" in name or "visual_proj" in name or "prompt" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    print(f"[Training] Trainable parameter count: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader), eta_min=1e-6)

    print("\n" + "-"*65)
    print(f"  Training for {num_epochs} epochs (Batches per epoch: {len(dataloader)})")
    print("-"*65)

    model.train()
    start_time = time.time()    # start from 1 for testing

    # START FROM EPOCH 1
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        step_count = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            import random
            cfg_lang_text = []
            for text in batch["lang_text"]:
                # CFG: 10%の確率でプロンプトを空文字("")にしてゼロトークン化する
                if random.random() < 0.1:
                    cfg_lang_text.append("")
                else:
                    cfg_lang_text.append(text)

            batch_gpu = {
                "rgb_obs": {
                    "rgb_static": batch["rgb_obs"]["rgb_static"].to(device),
                    "rgb_gripper": batch["rgb_obs"]["rgb_gripper"].to(device)
                },
                "lang_text": cfg_lang_text,
                "actions": batch["actions"].to(device)
            }

            obs_features = model.encode_observations(batch_gpu)
            act_loss, losses_dict = model.rf_loss(obs_features, batch_gpu["actions"])

            act_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += act_loss.item()
            step_count += 1
            
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar("Loss/train_batch", act_loss.item(), global_step)

            if (batch_idx + 1) % 150 == 0 or (batch_idx + 1) == len(dataloader):
                avg_b_loss = epoch_loss / step_count
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Batch [{batch_idx+1:04d}/{len(dataloader):04d}] | Loss: {avg_b_loss:.5f} | Elapsed: {elapsed/60:.1f}m")

        avg_loss = epoch_loss / max(1, step_count)
        print(f"--> Epoch {epoch:02d} Complete! Average Loss: {avg_loss:.5f}\n")

        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        ep_st_save_path = os.path.join(CHECKPOINT_SAVE_DIR, f"flower_hx5_finetuned_rescaled_ep{epoch:02d}.safetensors")
        cloned_state = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
        save_file(cloned_state, ep_st_save_path)
        print(f"[Save] Saved checkpoint for Epoch {epoch:02d} at {ep_st_save_path}")

    writer.close()

    # 学習済み重みの保存 (PyTorch pt & safetensors)
    pt_save_path = os.path.join(CHECKPOINT_SAVE_DIR, "flower_hx5_finetuned_rescaled.pt")
    st_save_path = os.path.join(CHECKPOINT_SAVE_DIR, "flower_hx5_finetuned_rescaled.safetensors")
    
    print(f"\n[Save] Saving fine-tuned model weights to: {pt_save_path}")
    torch.save(model.state_dict(), pt_save_path)

    # 共有テンソルをクローンしてsafetensors保存
    cloned_state = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    save_file(cloned_state, st_save_path)
    
    print("★ [Success] Fine-Tuning Completed and Saved Successfully!")
    print(f"  PyTorch Checkpoint: {pt_save_path}")
    print(f"  Safetensors Model: {st_save_path}\n")
    return st_save_path


if __name__ == "__main__":
    train_flower_adaptation(
        num_epochs=30,
        batch_size=8,
        lr=2.0e-5
    )
