#!/usr/bin/env python3
"""
FLOWER Fine-Tuning 100-Episode Dataset Generator with Full Domain Randomization
=============================================================================
【ランダム化の仕様】
1. アーム初期位置: 上空 x, y (±6cm), z (+25cm〜+35cm) のランダム配置
2. ブロック初期配置: 机上の x (±4cm), y (±4cm) のランダム移動
3. 左右配置のランダム入れ替え: 赤が左/ピンクが右、赤が右/ピンクが左 を50%ずつ網羅
4. 撮影カメラ: Static (200x200) + Wrist Camera (200x200, nearVal=0.08 遮蔽なし)
5. 収録エピソード: 計100エピソード（PINK 50回、RED 50回）
"""

import os
import sys
import json
import time
import math
import random
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path

WORKSPACE_DIR = "/home/ikeuchi/robotis"
FLOWER_DIR = "/home/ikeuchi/flower/flower_vla_calvin"
GRASP_AGENT_DIR = os.path.join(WORKSPACE_DIR, "vla/grasp_agent/active_size")

if FLOWER_DIR not in sys.path:
    sys.path.append(FLOWER_DIR)
if GRASP_AGENT_DIR not in sys.path:
    sys.path.append(GRASP_AGENT_DIR)

from pybullet_grasp_active_size import PyBulletGraspActiveSizeWrapper

URDF_PATH = os.path.join(WORKSPACE_DIR, "vla/data3/panda_hx5_right_custom.urdf")
DATA_SAVE_DIR = os.path.join(WORKSPACE_DIR, "vla/data42/episodes")


def capture_cameras(cid, robot_id, tcp_link_idx, gripper_cam_link_idx):
    """Static & Wrist カメラ生画像を撮影"""
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


def generate_single_episode(ep_idx: int, target_color: str = "pink", save_dir: str = DATA_SAVE_DIR):
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=cid)

    # 1. テーブル
    plane = p.loadURDF("plane.urdf", physicsClientId=cid)
    table = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.225], physicsClientId=cid),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 0.225], rgbaColor=[0.55, 0.40, 0.25, 1], physicsClientId=cid),
        basePosition=[0.6, 0.0, 0.225],
        physicsClientId=cid
    )

    box_half = [0.025, 0.025, 0.025]
    
    # 2. ブロック配置のランダム化（位置の散らばり ＋ 左右スワップ）
    swap_sides = (ep_idx % 4 >= 2)  # 50%の確率で左右を入れ替え
    
    rx1 = np.random.uniform(-0.035, 0.035)
    ry1 = np.random.uniform(-0.035, 0.035)
    rx2 = np.random.uniform(-0.035, 0.035)
    ry2 = np.random.uniform(-0.035, 0.035)

    base_y_pink = 0.08 if swap_sides else -0.07
    base_y_red = -0.07 if swap_sides else 0.08

    pink_pos = [0.635 + rx1, base_y_pink + ry1, 0.45 + box_half[2]]
    red_pos = [0.635 + rx2, base_y_red + ry2, 0.45 + box_half[2]]

    pink_box = p.createMultiBody(
        baseMass=0.01,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half, physicsClientId=cid),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=box_half, rgbaColor=[1.0, 0.3, 0.7, 1], physicsClientId=cid),
        basePosition=pink_pos,
        physicsClientId=cid
    )
    p.changeDynamics(pink_box, -1, lateralFriction=2.0, spinningFriction=0.1, restitution=0.0, physicsClientId=cid)

    red_box = p.createMultiBody(
        baseMass=0.01,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half, physicsClientId=cid),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=box_half, rgbaColor=[0.9, 0.1, 0.1, 1], physicsClientId=cid),
        basePosition=red_pos,
        physicsClientId=cid
    )
    p.changeDynamics(red_box, -1, lateralFriction=2.0, spinningFriction=0.1, restitution=0.0, physicsClientId=cid)

    # 3. ロボット初期化
    robot = p.loadURDF(URDF_PATH, [0, 0, 0.15], useFixedBase=True, physicsClientId=cid)
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        p.changeDynamics(robot, i, lateralFriction=2.5, spinningFriction=0.2, rollingFriction=0.05, restitution=0.0, physicsClientId=cid)

    wrapper = PyBulletGraspActiveSizeWrapper(physics_client_id=cid, robot_id=robot)
    tcp_link_idx = wrapper.link_indices.get("tcp", wrapper.arm_joint_indices[-1])
    gripper_cam_link_idx = wrapper.link_indices.get("gripper_cam", tcp_link_idx)

    target_pos = pink_pos if target_color == "pink" else red_pos
    target_body = pink_box if target_color == "pink" else red_box
    
    prompts = [
        f"pick up the {target_color} block",
        f"grasp the {target_color} block",
        f"lift the {target_color} cube",
        f"take the {target_color} object"
    ]
    lang_text = random.choice(prompts)

    # 把持目標姿勢 (arm_grasp_q)
    arm_grasp_q = wrapper.prepare_approach_and_preshape(box_pos=target_pos, box_half_extents=box_half)
    base_tcp_quat = p.getLinkState(robot, tcp_link_idx, physicsClientId=cid)[1]

    # 4. 高所初期位置のランダム化と指先オフセットの正確な計算
    # オフセットを正しく計算するため、一時的に把持姿勢（下向き）にセットする
    for i, idx in enumerate(wrapper.arm_joint_indices):
        p.resetJointState(robot, idx, arm_grasp_q[i], physicsClientId=cid)
    p.stepSimulation(physicsClientId=cid)

    thumb_idx = wrapper.link_indices["finger_end_r_link1"]
    middle_idx = wrapper.link_indices["finger_end_r_link3"]
    pb_tcp_state = p.getLinkState(robot, tcp_link_idx, physicsClientId=cid)
    base_tcp_pos = np.array(pb_tcp_state[0])
    pb_thumb = np.array(p.getLinkState(robot, thumb_idx, physicsClientId=cid)[0])
    pb_middle = np.array(p.getLinkState(robot, middle_idx, physicsClientId=cid)[0])
    base_midpoint_offset = (pb_thumb * (2.0/3.0) + pb_middle * (1.0/3.0)) - base_tcp_pos

    # 4. 高所初期位置のランダム化 (X,Yにズレあり、Z=0.28)
    # ターゲットではなく、テーブル上の絶対的な固定中点を基準とする（プロンプトの力を測るための中立位置）
    neutral_pos_mid = np.array([0.635, 0.025, 0.45 + box_half[2]])

    # スタート位置のバリエーション (Domain Randomization)
    start_rx = np.random.uniform(-0.01, 0.01)
    start_ry = np.random.uniform(-0.01, 0.01)
    high_start_mid = neutral_pos_mid + np.array([start_rx, start_ry, 0.28])

    target_pos_mid = np.array(target_pos)
    q_ik_high = p.calculateInverseKinematics(
        robot, tcp_link_idx, high_start_mid - base_midpoint_offset,
        targetOrientation=base_tcp_quat,
        maxNumIterations=200, residualThreshold=1e-5, physicsClientId=cid
    )
    arm_high_q = np.array([q_ik_high[idx] for idx in wrapper.arm_joint_indices], dtype=np.float32)

    for i, idx in enumerate(wrapper.arm_joint_indices):
        p.resetJointState(robot, idx, arm_high_q[i], physicsClientId=cid)
        p.setJointMotorControl2(robot, idx, controlMode=p.POSITION_CONTROL, targetPosition=arm_high_q[i], force=500.0, physicsClientId=cid)

    for _ in range(15):
        p.stepSimulation(physicsClientId=cid)

    ep_static_imgs = []
    ep_wrist_imgs = []
    ep_actions = []
    ep_robot_obs = []

    # Phase 1A: 真上への移動 (gripper = +1.0)
    align_steps = 15
    start_pos = high_start_mid
    perfect_above_pos = target_pos_mid + np.array([0.0, 0.0, 0.28])
    for step in range(align_steps):
        rgb_static, rgb_wrist, curr_tcp_pos, curr_tcp_orn = capture_cameras(cid, robot, tcp_link_idx, gripper_cam_link_idx)
        alpha = (step + 1) / float(align_steps)
        perfect_target_pos = (1.0 - alpha) * start_pos + alpha * perfect_above_pos
        
        # エキスパートのアクション: 現在の(ノイズを含んだ)位置から、本来の完璧な目標位置への差分ベクトル
        # 指先の目標位置(perfect_target_pos)と、現在の手首位置(curr_tcp_pos)の差分にオフセットを考慮
        expert_d_pos = perfect_target_pos - (np.array(curr_tcp_pos) + base_midpoint_offset)
        d_pos_scaled = expert_d_pos * 100.0
        action_7d = np.array([d_pos_scaled[0], d_pos_scaled[1], d_pos_scaled[2], 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # データ収集用のノイズ注入 (Light DAgger)
        # 次のステップのために、わざと軌道を少し(±3mm)ズラした位置に移動させる
        noise = np.random.uniform(-0.015, 0.015, size=3)
        noisy_target_pos = perfect_target_pos + noise
        
        q_ik = p.calculateInverseKinematics(
            robot, tcp_link_idx, noisy_target_pos - base_midpoint_offset,
            targetOrientation=base_tcp_quat,
            maxNumIterations=100, residualThreshold=1e-4, physicsClientId=cid
        )
        for i, idx in enumerate(wrapper.arm_joint_indices):
            p.setJointMotorControl2(robot, idx, controlMode=p.POSITION_CONTROL, targetPosition=q_ik[idx], force=500.0, physicsClientId=cid)
        for _ in range(3):
            p.stepSimulation(physicsClientId=cid)
        
        ep_static_imgs.append(rgb_static)
        ep_wrist_imgs.append(rgb_wrist)
        ep_actions.append(action_7d)
        ep_robot_obs.append(list(curr_tcp_pos) + list(curr_tcp_orn))

    # Phase 1B: 垂直降下 (gripper = +1.0)
    drop_steps = 28
    for step in range(drop_steps):
        rgb_static, rgb_wrist, curr_tcp_pos, curr_tcp_orn = capture_cameras(cid, robot, tcp_link_idx, gripper_cam_link_idx)
        alpha = (step + 1) / float(drop_steps)
        perfect_target_pos = (1.0 - alpha) * perfect_above_pos + alpha * target_pos_mid
        
        # エキスパートのアクション: 現在の(ノイズを含んだ)位置から、本来の完璧な目標位置への差分ベクトル
        expert_d_pos = perfect_target_pos - (np.array(curr_tcp_pos) + base_midpoint_offset)
        d_pos_scaled = expert_d_pos * 100.0
        action_7d = np.array([d_pos_scaled[0], d_pos_scaled[1], d_pos_scaled[2], 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # データ収集用のノイズ注入 (Light DAgger)
        noise = np.random.uniform(-0.015, 0.015, size=3)
        noisy_target_pos = perfect_target_pos + noise
        
        q_ik = p.calculateInverseKinematics(
            robot, tcp_link_idx, noisy_target_pos - base_midpoint_offset,
            targetOrientation=base_tcp_quat,
            maxNumIterations=100, residualThreshold=1e-4, physicsClientId=cid
        )
        for i, idx in enumerate(wrapper.arm_joint_indices):
            p.setJointMotorControl2(robot, idx, controlMode=p.POSITION_CONTROL, targetPosition=q_ik[idx], force=500.0, physicsClientId=cid)
        for _ in range(3):
            p.stepSimulation(physicsClientId=cid)
        
        ep_static_imgs.append(rgb_static)
        ep_wrist_imgs.append(rgb_wrist)
        ep_actions.append(action_7d)
        ep_robot_obs.append(list(curr_tcp_pos) + list(curr_tcp_orn))

    # Phase 2: 把持信号の送信 (gripper = -1.0)
    # ここでエピソードを打ち切ることで、VLAにノイズだらけのSACの動きを学習させないようにする
    for grasp_step in range(3):
        rgb_static, rgb_wrist, curr_tcp_pos, curr_tcp_orn = capture_cameras(cid, robot, tcp_link_idx, gripper_cam_link_idx)
        action_7d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        ep_static_imgs.append(rgb_static)
        ep_wrist_imgs.append(rgb_wrist)
        ep_actions.append(action_7d)
        ep_robot_obs.append(list(curr_tcp_pos) + list(curr_tcp_orn))
        p.stepSimulation(physicsClientId=cid)

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"episode_{ep_idx:04d}_{target_color}.npz")
    prompt = lang_text
    np.savez_compressed(
        save_path,
        rgb_static=np.array(ep_static_imgs, dtype=np.uint8),
        rgb_gripper=np.array(ep_wrist_imgs, dtype=np.uint8),
        actions=np.array(ep_actions, dtype=np.float32),
        robot_obs=np.array(ep_robot_obs, dtype=np.float32),
        prompt=prompt
    )

    # メタデータの保存
    ep_metadata = {
        "episode_id": os.path.basename(save_path),
        "target_color": target_color,
        "prompt": prompt,
        "pink_pos": list(pink_pos),
        "red_pos": list(red_pos),
        "hand_start_pos": list(high_start_mid)
    }
    metadata_file = os.path.join(DATA_SAVE_DIR, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            try:
                metadata_list = json.load(f)
            except:
                metadata_list = []
    else:
        metadata_list = []
    metadata_list.append(ep_metadata)
    with open(metadata_file, "w") as f:
        json.dump(metadata_list, f, indent=4)

    p.disconnect(cid)
    print(f"[{ep_idx:03d}/10] {target_color.upper()} (Swap={swap_sides}): {len(ep_actions)} steps -> OK")
    return save_path


def main():
    print("\n" + "="*65)
    print("  Generating 500 Multi-Modal Episodes for FLOWER Fine-Tuning")
    print("="*65)
    
    total_episodes = 500
    for i in range(total_episodes):
        target = "pink" if i % 2 == 0 else "red"
        npz_path = generate_single_episode(ep_idx=i+1, target_color=target)

    print("\n★ 300 Episodes Successfully Generated and Saved to:")
    print(f"  {DATA_SAVE_DIR}\n")



if __name__ == "__main__":
    main()
