#!/usr/bin/env python3
"""
Test Fine-Tuned FLOWER VLA with Complete Grasping Rollout
=========================================================
1. FLOWER VLA による高所からの滑らかな自律下降 (APPROACH)
2. 目標到達・Close信号検知による SAC 多指把持AI への自律バトンタッチ (GRASP)
3. 5本指による完全包み込み把持＆上空へのリフトアップ (LIFT)
の全フェーズを完全録画します。
"""

import os
import sys
import time
import math
import cv2
import torch
import numpy as np
import pybullet as p
import pybullet_data
from omegaconf import OmegaConf
from safetensors.torch import load_file

WORKSPACE_DIR = "/home/ikeuchi/robotis"
FLOWER_DIR = "/home/ikeuchi/flower/flower_vla_calvin"
GRASP_AGENT_DIR = os.path.join(WORKSPACE_DIR, "vla/grasp_agent/active_size")

if FLOWER_DIR not in sys.path:
    sys.path.append(FLOWER_DIR)
if GRASP_AGENT_DIR not in sys.path:
    sys.path.append(GRASP_AGENT_DIR)

from flower.models.flower import FLOWERVLA
from pybullet_grasp_active_size import PyBulletGraspActiveSizeWrapper

URDF_PATH = os.path.join(WORKSPACE_DIR, "vla/data3/panda_hx5_right_custom.urdf")
FINETUNED_MODEL_PATH = os.path.join(WORKSPACE_DIR, "vla/data42/checkpoints/flower_hx5_finetuned_rescaled/flower_hx5_finetuned_rescaled_ep30.safetensors")
CONFIG_PATH = os.path.join(FLOWER_DIR, "pretrained/flower_calvin_d/config.yaml")
OUT_VIDEO = os.path.join(WORKSPACE_DIR, "vla/data42/finetuned_flower_grasp_demo_ep30_pink_block.mp4")


def capture_cameras(cid, robot_id, tcp_link_idx, gripper_cam_link_idx):
    # 1. Static Camera (200x200)
    cam_pos = [-0.15, -0.60, 0.78]
    target_pos = [0.60, 0.0, 0.45]
    v_mat = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1], physicsClientId=cid)
    p_mat = p.computeProjectionMatrixFOV(55, 1.0, 0.1, 2.5, physicsClientId=cid)
    _, _, rgb_static, _, _ = p.getCameraImage(200, 200, v_mat, p_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=cid)
    rgb_static_np = np.array(rgb_static, dtype=np.uint8).reshape((200, 200, 4))[:, :, :3]

    # 2. Wrist Camera (200x200, nearVal=0.08)
    camera_ls = p.getLinkState(robot_id, gripper_cam_link_idx, computeForwardKinematics=True, physicsClientId=cid)
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = np.array(p.getMatrixFromQuaternion(camera_orn)).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]

    view_matrix = p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z, physicsClientId=cid)
    proj_matrix = p.computeProjectionMatrixFOV(fov=75, aspect=1.0, nearVal=0.001, farVal=2.0, physicsClientId=cid)

    img = p.getCameraImage(200, 200, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=cid)
    rgb_wrist_np = np.array(img[2], dtype=np.uint8).reshape((200, 200, 4))[:, :, :3]

    tcp_pos, tcp_orn = p.getLinkState(robot_id, tcp_link_idx, physicsClientId=cid)[:2]
    return rgb_static_np, rgb_wrist_np, tcp_pos, tcp_orn


def main():
    print("\n" + "="*65)
    print("  Testing Fine-Tuned FLOWER VLA Autonomous Grasping Pipeline")
    print("="*65)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # 1. モデルロード
    cfg = OmegaConf.load(CONFIG_PATH)
    model_cfg = cfg.model
    if "_target_" in model_cfg: del model_cfg["_target_"]
    if "_recursive_" in model_cfg: del model_cfg["_recursive_"]
    model_cfg["pretrained_model_path"] = None
    model_cfg["load_pretrained"] = False
    model_cfg["query_seq_len"] = 120

    model = FLOWERVLA(**model_cfg)
    
    state_dict = load_file(FINETUNED_MODEL_PATH)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. PyBullet 環境
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=cid)

    plane = p.loadURDF("plane.urdf", physicsClientId=cid)
    table = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.225], physicsClientId=cid),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.225], rgbaColor=[0.55, 0.40, 0.25, 1], physicsClientId=cid),
        basePosition=[0.6, 0.0, 0.225],
        physicsClientId=cid
    )

    box_half = [0.025, 0.025, 0.025]
    pink_pos = [0.635, -0.06, 0.45 + box_half[2]]
    pink_box = p.createMultiBody(
        baseMass=0.01,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half, physicsClientId=cid),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=box_half, rgbaColor=[1.0, 0.3, 0.7, 1], physicsClientId=cid),
        basePosition=pink_pos,
        physicsClientId=cid
    )
    p.changeDynamics(pink_box, -1, lateralFriction=2.5, spinningFriction=0.2, restitution=0.0, physicsClientId=cid)

    red_pos = [0.635, 0.08, 0.45 + box_half[2]]
    red_box = p.createMultiBody(
        baseMass=0.01,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.025, height=0.05, physicsClientId=cid),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.05, rgbaColor=[0.9, 0.1, 0.1, 1], physicsClientId=cid),
        basePosition=red_pos,
        physicsClientId=cid
    )
    p.changeDynamics(red_box, -1, lateralFriction=2.5, spinningFriction=0.2, restitution=0.0, physicsClientId=cid)

    robot = p.loadURDF(URDF_PATH, [0, 0, 0.15], useFixedBase=True, physicsClientId=cid)
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        p.changeDynamics(robot, i, lateralFriction=2.5, spinningFriction=0.2, rollingFriction=0.05, restitution=0.0, physicsClientId=cid)

    wrapper = PyBulletGraspActiveSizeWrapper(physics_client_id=cid, robot_id=robot)
    tcp_link_idx = wrapper.link_indices.get("tcp", wrapper.arm_joint_indices[-1])
    gripper_cam_link_idx = wrapper.link_indices.get("gripper_cam", tcp_link_idx)

    # 初期プレシェイプ & 高所スタート (+28cm)
    arm_grasp_q = wrapper.prepare_approach_and_preshape(box_pos=pink_pos, box_half_extents=box_half)
    base_tcp_quat = p.getLinkState(robot, tcp_link_idx, physicsClientId=cid)[1]

    thumb_idx = wrapper.link_indices["finger_end_r_link1"]
    middle_idx = wrapper.link_indices["finger_end_r_link3"]
    pb_tcp_state = p.getLinkState(robot, tcp_link_idx, physicsClientId=cid)
    base_tcp_pos = np.array(pb_tcp_state[0])
    pb_thumb = np.array(p.getLinkState(robot, thumb_idx, physicsClientId=cid)[0])
    pb_middle = np.array(p.getLinkState(robot, middle_idx, physicsClientId=cid)[0])
    base_midpoint_offset = (pb_thumb * (2.0/3.0) + pb_middle * (1.0/3.0)) - base_tcp_pos

    # 学習データに合わせ、2つのブロック（pink_posとred_pos）の完全な中間点を初期位置とする
    mid_pos = (np.array(pink_pos) + np.array(red_pos)) / 2.0
    high_start_mid = mid_pos + np.array([0.0, 0.0, 0.28])
    q_ik_high = p.calculateInverseKinematics(
        robot, tcp_link_idx, high_start_mid - base_midpoint_offset,
        targetOrientation=base_tcp_quat,
        maxNumIterations=200, residualThreshold=1e-5, physicsClientId=cid
    )
    for i, idx in enumerate(wrapper.arm_joint_indices):
        p.resetJointState(robot, idx, q_ik_high[idx], physicsClientId=cid)
        p.setJointMotorControl2(robot, idx, controlMode=p.POSITION_CONTROL, targetPosition=q_ik_high[idx], force=500.0, physicsClientId=cid)

    for _ in range(15):
        p.stepSimulation(physicsClientId=cid)

    prompt = "pick up the pink block"
    print(f"\n[Goal Prompt]: '{prompt}'")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(OUT_VIDEO, fourcc, 12.0, (640, 480))

    def write_frame(rgb_s, rgb_w, step_num, phase_text, is_grasp_phase, dz_val, grip_val):
        canvas = np.full((480, 640, 3), 30, dtype=np.uint8)
        
        # Static Camera
        bgr_s = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR)
        canvas[70:450, 20:400] = cv2.resize(bgr_s, (380, 380), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(canvas, (20, 70), (400, 450), (200, 200, 200), 2)
        cv2.putText(canvas, "Static Camera (Fine-Tuned FLOWER)", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Wrist Camera
        bgr_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2BGR)
        canvas[70:260, 430:620] = cv2.resize(bgr_w, (190, 190), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(canvas, (430, 70), (620, 260), (0, 255, 255), 2)
        cv2.putText(canvas, "Wrist Camera (Eye-in-Hand)", (435, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # Status Panel
        phase_col = (0, 100, 255) if is_grasp_phase else (0, 255, 0)
        cv2.rectangle(canvas, (430, 280), (620, 450), (45, 45, 45), -1)
        cv2.rectangle(canvas, (430, 280), (620, 450), phase_col, 2)
        cv2.putText(canvas, "Fine-Tuned FLOWER", (440, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(canvas, f"Step: {step_num:02d}", (440, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cv2.putText(canvas, "Phase:", (440, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(canvas, phase_text, (440, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.40, phase_col, 1)
        cv2.putText(canvas, f"dz: {dz_val:+.4f}", (440, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(canvas, f"Gripper: {grip_val:+.2f}", (440, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.putText(canvas, "Fine-Tuned FLOWER Multi-Finger Grasp Pipeline", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, f"Prompt: '{prompt}'", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 100), 1)
        video_out.write(canvas)

    # 1. FLOWER VLA 自律アプローチフェーズ (Closed-Loop)
    print("\n--- Phase 1: FLOWER VLA Autonomous Approach (Closed-Loop) ---")
    approach_steps = 150
    
    tcp_current = np.array(p.getLinkState(robot, tcp_link_idx, physicsClientId=cid)[0], dtype=np.float64)
    grip_history = []
    handover_step = approach_steps
    
    for step in range(approach_steps):
        rgb_s, rgb_w, curr_tcp_pos, _ = capture_cameras(cid, robot, tcp_link_idx, gripper_cam_link_idx)
        
        t_static = torch.from_numpy(rgb_s).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        t_wrist = torch.from_numpy(rgb_w).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            pred_chunk = model({"rgb_obs": {"rgb_static": t_static, "rgb_gripper": t_wrist}}, {"lang_text": prompt}).cpu().numpy()[0]
        
        act = pred_chunk[0]
        d_pos = act[:3] / 100.0
        dz = d_pos[2]
        grip = act[6]

        grip_history.append(grip)
        if len(grip_history) > 5:
            grip_history.pop(0)
        grip_ma = sum(grip_history) / len(grip_history)

        tcp_current += d_pos
        tcp_current[2] = max(0.48, tcp_current[2])  # 机上面へのめり込み防止

        q_ik = p.calculateInverseKinematics(
            robot, tcp_link_idx, tcp_current,
            targetOrientation=base_tcp_quat,
            maxNumIterations=100, residualThreshold=1e-4, physicsClientId=cid
        )
        for i, idx in enumerate(wrapper.arm_joint_indices):
            p.setJointMotorControl2(robot, idx, controlMode=p.POSITION_CONTROL, targetPosition=q_ik[idx], force=500.0, physicsClientId=cid)

        for _ in range(3):
            p.stepSimulation(physicsClientId=cid)

        write_frame(rgb_s, rgb_w, step + 1, "1. FLOWER Approach", False, dz, grip)
        
        if len(grip_history) >= 5 and grip_ma <= -0.7:
            print(f"★ CLOSE SIGNAL TRIGGERED at Step {step}! grip_ma={grip_ma:+.2f}, dz={dz:+.4f}")
            handover_step = step + 1
            break

    # 2. Close 信号出力 & SAC 多指把持・リフトアップフェーズ
    print("\n--- Phase 2: SAC Multi-Finger Grasp & Lift ---")
    
    # オラクル情報への依存を減らすため、RLへの制御バトンタッチ時点での
    # 親指と中指を結ぶ点を物体の中心(XYZ)として推定・上書きする
    curr_thumb = np.array(p.getLinkState(robot, thumb_idx, physicsClientId=cid)[0])
    curr_middle = np.array(p.getLinkState(robot, middle_idx, physicsClientId=cid)[0])
    estimated_box_pos = curr_thumb * (2.0/3.0) + curr_middle * (1.0/3.0)
    wrapper.current_box_pos = estimated_box_pos.tolist()

    current_arm_q = np.zeros(7, dtype=np.float32)
    for i, idx in enumerate(wrapper.arm_joint_indices):
        current_arm_q[i] = p.getJointState(robot, idx, physicsClientId=cid)[0]
    wrapper.arm_grasp_q = np.copy(current_arm_q)
    
    wrapper.step_counter = 0
    wrapper.previous_action = np.zeros(2, dtype=np.float32)
    wrapper.is_ready_to_lift = False
    wrapper.grasp_stable_count = 0
    wrapper.lift_start_step = 0
    
    for grasp_step in range(wrapper.max_steps):
        rgb_s, rgb_w, _, _ = capture_cameras(cid, robot, tcp_link_idx, gripper_cam_link_idx)
        info = wrapper.step_grasp()

        curr_box_z = p.getBasePositionAndOrientation(red_box, physicsClientId=cid)[0][2]
        lift = curr_box_z - (0.45 + box_half[2])

        write_frame(rgb_s, rgb_w, handover_step + grasp_step + 1, "2. SAC Grasp & Lift", True, 0.0, -1.0)

        if info["is_ready_to_lift"] and lift > 0.025 and grasp_step > 30:
            print(f"★ [Step {handover_step + grasp_step + 1}] Successfully Lifted by {lift*100:.1f} cm!")
            # 完了状態で15フレーム保持
            for _ in range(15):
                rgb_s, rgb_w, _, _ = capture_cameras(cid, robot, tcp_link_idx, gripper_cam_link_idx)
                write_frame(rgb_s, rgb_w, handover_step + grasp_step + 1, "★ Lift Success!", True, 0.0, -1.0)
            break

    video_out.release()
    p.disconnect(cid)
    print(f"\n★ Complete Video Successfully Saved to:\n  {OUT_VIDEO}\n")


if __name__ == "__main__":
    main()
