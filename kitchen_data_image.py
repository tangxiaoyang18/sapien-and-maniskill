import gymnasium as gym
import mani_skill.envs
import numpy as np
import h5py
import torch
import os
import time
from PIL import Image
import math
import sapien
from mani_skill.utils.wrappers import CPUGymWrapper

def setup_cameras(scene, num_cameras=8, radius=1.5, height=1):
    """设置多个相机围绕场景"""
    cameras = []
    width, height_res = 640, 480
    robot_position=[3.0000,-2.0250,0.0000]
    target_point = robot_position + np.array([0, 0, 1])
#---------------------------------------------------------------------
    top_cam_pos = robot_position + np.array([0, 0, 3])
    forward = (robot_position-top_cam_pos)/np.linalg.norm(top_cam_pos)

    left = np.cross(np.array([1, 0, 0]), forward)
    left = left / np.linalg.norm(left)

    up = np.cross(forward, left)
    up = up/np.linalg.norm(up)

    mat44=np.eye(4)
    mat44[:3,:3] = np.stack([forward,left,up],axis=1)
    mat44[:3,3] = top_cam_pos

    camera = scene.add_camera(
        name="top_camera",
        pose =sapien.Pose(mat44),
        width=width,
        height=height_res,
        fovy=np.deg2rad(60),
        near=0.1,
        far=100,
    )
    #camera.set_local_pose(sapien.Pose([0, 0, 3], [1, 0, 0, 0]))
    cameras.append(camera)

    # 添加环绕相机
    for i in range(num_cameras):
        angle = 2 * math.pi * i / num_cameras
        # 以机器人为中心的环绕坐标
        x = robot_position[0] + radius * math.cos(angle)
        y = robot_position[1] + radius * math.sin(angle)
        z = robot_position[2] + height

        cam_pos = np.array([x, y, z])

        # 计算相机朝向：从相机位置指向机器人目标点
        forward = (target_point - cam_pos)
        forward = forward / np.linalg.norm(forward)  # 归一化方向向量

        left = np.cross(np.array([0, 0, 1]), forward)
        left = left / np.linalg.norm(left)

        up = np.cross(forward, left)
        up = up / np.linalg.norm(up)

        # 构建旋转矩阵并转换为四元数
        mat44 = np.eye(4)
        mat44[:3,:3] = np.stack([forward, left, up],axis=1)
        mat44[:3, 3] = cam_pos

        camera = scene.add_camera(
            name=f"camera_{i}",
            pose=sapien.Pose(mat44),
            width=width,
            height=height_res,
            fovy=np.deg2rad(80),
            near=0.1,
            far=100
        )
        #camera.set_local_pose(sapien.Pose([x, y, z]))
        cameras.append(camera)

    return cameras


def capture_and_save_images(cameras, save_dir, step):
    """从所有相机捕获图像并保存到统一目录"""
    for camera in cameras:
        # 更新相机观察
        camera.take_picture()

        # 获取图像数据（List[torch.Tensor]）
        tensor_list = camera.get_picture("Color")

        if len(tensor_list) > 0:
            # 处理张量并转换为numpy数组
            rgba_tensor = tensor_list[0].cpu() if tensor_list[0].is_cuda else tensor_list[0]
            rgba = rgba_tensor.numpy()

            # 处理4维批次维度
            if rgba.ndim == 4:
                rgba = np.squeeze(rgba, axis=0)

            # 处理3维情况
            if rgba.ndim == 3:
                # 转换通道顺序 (C, H, W) -> (H, W, C)
                if rgba.shape[0] in [3, 4]:
                    rgba = rgba.transpose(1, 2, 0)

                if rgba.shape[2] == 4:
                    rgb = rgba[..., :3]
                    rgb = (rgb * 255).astype(np.uint8)

                    # 保存图像到统一目录，文件名包含相机名称和步骤号
                    filename = f"step_{step:06d}_{camera.name}.png"
                    save_path = os.path.join(save_dir, filename)
                    img = Image.fromarray(rgb)
                    img.save(save_path)
                else:
                    print(f"警告: 相机 {camera.name} 通道数不正确，形状为 {rgba.shape}")
            else:
                print(f"警告: 相机 {camera.name} 维度不正确，维度为 {rgba.ndim}")
        else:
            print(f"警告: 相机 {camera.name} 未返回图像数据")



# 创建环境
num_envs = 1  # 为了简化数据记录，使用单个环境
env = gym.make("RoboCasaKitchen-v1", num_envs=num_envs, render_mode="human", obs_mode="state")
obs, _ = env.reset()

# 获取场景和机器人
scene = env.unwrapped.scene
robot = env.unwrapped.agent.robot

# 设置多个相机
cameras = setup_cameras(scene)

# 准备HDF5文件
timestamp = time.strftime("%Y%m%d_%H%M%S")
h5_filename = f"data/jointstate/robot_data_{timestamp}.h5"
h5_file = h5py.File(h5_filename, "w")

# 创建数据集
max_steps = 10000
joint_positions = h5_file.create_dataset(
    "joint_positions",
    (max_steps, len(robot.get_active_joints())),
    maxshape=(None, len(robot.get_active_joints())),
    dtype='float32',
    compression="gzip"
)

actions = h5_file.create_dataset(
    "actions",
    (max_steps, env.action_space.shape[0]),
    maxshape=(None, env.action_space.shape[0]),
    dtype='float32',
    compression="gzip"
)

dones = h5_file.create_dataset(
    "dones",
    (max_steps,),
    maxshape=(None,),
    dtype='bool'
)

timestamps = h5_file.create_dataset(
    "timestamps",
    (max_steps,),
    maxshape=(None,),
    dtype='float64'
)

# 记录数据
step_count = 0
start_time = time.time()

try:
    done = False
    while not done and step_count < max_steps:
        # 生成随机动作
        action = env.action_space.sample()

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = False

        # 渲染环境
        env.render()
        #if step_count % 100 == 0:
            #capture_and_save_images(cameras, "data/image", step_count)


        joint_positions[step_count] = robot.get_qpos()
        actions[step_count] = action
        dones[step_count] = done
        timestamps[step_count] = time.time() - start_time

        step_count += 1

        # 如果环境结束，重置环境
        if done:
            obs, _ = env.reset()
            done = False

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # 调整数据集大小以匹配实际步数
    joint_positions.resize((step_count, len(robot.get_active_joints())))
    actions.resize((step_count, env.action_space.shape[0]))
    dones.resize((step_count,))
    timestamps.resize((step_count,))

    # 关闭HDF5文件和环境
    h5_file.close()
    env.close()