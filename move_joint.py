import sapien
import numpy as np
from PIL import Image
import math
import os
import h5py
import time

def set_camera(near,far,width,height,scene,cam_pos,look_at=None,fovy=35):#一个看向原点的相机
    if look_at is None:
        look_at = np.array([0, 0, 0])  # 默认看向原点

    forward = (look_at - cam_pos) / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    if np.linalg.norm(left) < 1e-6:  # 处理相机正对上方或下方的情况
        left = np.array([1, 0, 0])
    else:
        left = left / np.linalg.norm(left)

    up = np.cross(forward, left)
    up = up / np.linalg.norm(up)

    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(fovy),
        near=near,
        far=far,
    )
    camera.entity.set_pose(sapien.Pose(mat44))
    return camera

def setup_cameras(scene, num_cameras=8, radius=2.5, height=1.0):

    cameras = []
    width, height_res = 640, 480

    cameras.append(set_camera(
        near=0.1, 
        far=100, 
        width=width, 
        height=height_res, 
        scene=scene, 
        cam_pos=np.array([0, 0, 3]), 
        look_at=np.array([0, 0, 0]),
        fovy=60
    ))#设置顶部相机

    for i in range(num_cameras):
        angle = 2 * math.pi * i / num_cameras
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        
        cameras.append(set_camera(
            0.1, 100, width, height_res, scene,
            cam_pos=np.array([x, y, z]),
            look_at=np.array([0, 0, height/2]),
            fovy=45
        ))#环绕相机
    
    for i in range(num_cameras//2):
        angle = 2 * math.pi * i / (num_cameras//2)
        x = radius * 0.7 * math.cos(angle)
        y = radius * 0.7 * math.sin(angle)
        z = 0.3  # 较低的高度
        
        cameras.append(set_camera(
            0.1, 100, width, height_res, scene,
            cam_pos=np.array([x, y, z]),
            look_at=np.array([0, 0, 0.5]),
            fovy=55
        ))#较低环绕相机

    return cameras

def move_joint():

    figure_save_path = "data/image"
    hdf5_save_path = "data/jointstate"

    # Create a unique timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    #Set up the engine,renderer and scene
    scene = sapien.Scene()
    scene.set_timestep(1 / 2000.0)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Load URDF,使用Kinova Jaco2 机械臂
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True #fix_root_link是指固定robot的根链接
    robot = loader.load("assets/jaco2/jaco2.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))#位置和旋转姿态四元数
    #此时由于重力，机械臂会下垂

    # Set joint positions
    arm_zero_qpos = [0, 3.14, 0, 3.14, 0, 3.14, 0] #手臂零位姿态（7个关节，7自由度）
    gripper_init_qpos = [0, 0, 0, 0, 0, 0] #夹爪零位姿态（6自由度，3个爪，2个关节）
    zero_qpos = arm_zero_qpos + gripper_init_qpos #机械臂和夹爪的初始位置（13自由度）
    robot.set_qpos(zero_qpos)#设置初始状态
    arm_target_qpos = [4.71, 2.84, 0.0, 0.75, 4.62, 4.48, 4.88]
    target_qpos = arm_target_qpos + [1, 1, 1, 1, 1, 1]

    active_joints = robot.get_active_joints()

    for joint_idx, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=20, damping=5, force_limit=1000, mode="force")
        joint.set_drive_target(target_qpos[joint_idx])
    
    cameras = setup_cameras(scene, num_cameras=8, radius=2.5, height=1.2)

    h5_filename = os.path.join(hdf5_save_path, f"robot_data_{timestamp}.h5")
    h5_file = h5py.File(h5_filename, "w")  # 正确的语法
    
    max_frames = 10000
    joint_positions = h5_file.create_dataset(
        "joint_positions", 
        (max_frames, len(active_joints)), 
        maxshape=(None, len(active_joints)),
        dtype='float32',
        compression="gzip"
    )
    timestamps = h5_file.create_dataset(
        "timestamps",
        (max_frames,),
        maxshape=(None,),
        dtype='float64'
    )
    frame_counter = 0

    frcounter = 0
    start_time = time.time()
    
    try:
        while not viewer.closed:
            frcounter += 1
            for _ in range(4):  # render every 4 steps
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf) #qf:设置关节力,返回的顺序同robot.get_joints()
                scene.step()
            scene.update_render()
            viewer.render()
            
            if frcounter % 10 == 0:
                for i, camera in enumerate(cameras):
                    camera.take_picture()
                    rgba = camera.get_picture("Color")
                    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
                    rgba_pil = Image.fromarray(rgba_img)
                    rgba_pil.save(os.path.join(figure_save_path, f"camera_{i}_frame_{frcounter}.png"))

                if frame_counter >= joint_positions.shape[0]:
                    joint_positions.resize((joint_positions.shape[0] + 1000, len(active_joints)))
                    timestamps.resize((timestamps.shape[0] + 1000,))
                
                joint_positions[frame_counter] = robot.get_qpos()
                timestamps[frame_counter] = time.time() - start_time
                frame_counter += 1

    finally:
        # 调整HDF5数据集大小并关闭
        joint_positions.resize((frame_counter, len(active_joints)))
        timestamps.resize((frame_counter,))
        h5_file.close()
        
        


def main():
    move_joint()
   
if __name__ == "__main__":
    main()