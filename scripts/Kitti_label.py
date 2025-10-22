import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
import numpy as np
import os
from PIL import Image
# import open3d as o3d
import tqdm

def to_homogeneous(matrix_3x4):
    """
    Converts a 3x4 matrix to a 4x4 homogeneous matrix by appending [0,0,0,1] as the last row.
    
    Parameters:
        matrix_3x4 (numpy.ndarray): A 3x4 matrix.
        
    Returns:
        numpy.ndarray: A 4x4 homogeneous matrix.
    """
    return np.vstack((matrix_3x4, np.array([0, 0, 0, 1])))

def read_calibration_file(file_path):
    """
    Reads a calibration file in .txt format and converts each line into a key with a 3x4 NumPy matrix.
    
    Expected file format:
        P0: <12 floating-point numbers>
        P1: <12 floating-point numbers>
        P2: <12 floating-point numbers>
        P3: <12 floating-point numbers>
        Tr: <12 floating-point numbers>
        
    Parameters:
        file_path (str): Path to the calibration file.
      
    Returns:
        dict: A dictionary with keys (e.g., 'P0', 'P1', etc.) mapping to 3x4 NumPy arrays.
    """
    calibration_data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.split()
            key = parts[0].rstrip(':')  # Remove the trailing colon
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError as e:
                print(f"Error converting values for key {key}: {e}")
                continue
            
            if len(values) != 12:
                print(f"Warning: Expected 12 elements for key '{key}', but got {len(values)}.")
                continue
            
            matrix = np.array(values).reshape((3, 4))
            calibration_data[key] = matrix
            
    return calibration_data

def compute_camera_extrinsics(calib_data, data_type = "KRt"):
    """
    Computes each camera's extrinsic parameters in the body coordinate system.
    
    For P0-P3:
      - Each entry is given as a 3x4 matrix [K|t], where:
         - K is the 3x3 intrinsic matrix.
         - t is the translation from camera P0 to the current camera (with no rotation).
      - 'Tr' is the transformation [R|t] from Body to camera P0.
    
    The transformation from Body to camera i is computed as:
        T_cam_body = T_{P_i}^{P_0} * T_{cam0}^{Body}
    where:
        T_{P_i}^{P_0} = [I_{3x3}  t_i; 0  1]
        T_{cam0}^{Body} = to_homogeneous(Tr)
    
    Returns a dictionary with each camera key (e.g., "P0") mapping to a dictionary with:
        - "K": the 3x3 intrinsic matrix.
        - "T_cam_body": the 4x4 transformation matrix from Body to the camera coordinate system.
    
    Parameters:
        calib_data (dict): Dictionary returned by read_calibration_file.
    
    Returns:
        dict: Dictionary with keys for each camera.
    """
    # Retrieve the transformation from Body to camera P0.
    if "Tr" not in calib_data:
        raise KeyError("Tr (Body to camera P0 transformation) not found in calibration data.")
    
    T_body_cam0 = to_homogeneous(calib_data["Tr"])
    # T_body_cam0 = np.array([[0.0, -1.0, 0.0, 0.0],
    #                         [0.0, 0.0, -1.0,0.0],
    #                         [1.0,0.0,0.0,0.0],
    #                         [0.0, 0.0,0.0, 1.0]])
    
    camera_data = {}
    # Process cameras P0-P3.
    for key, matrix in calib_data.items():
        if key.startswith("P"):
            # Split into intrinsic matrix K and translation vector t_rel (from camera P0 to current camera)
            K = matrix[:, :3]
            t_rel = matrix[:, 3].reshape((3, 1))  # Ensure it's a column vector
            if data_type == "KRt":
              t_rel = np.linalg.inv(K) @ t_rel
            # Build the homogeneous transformation from camera P0 to camera i.
            T_p0_pi = np.vstack((np.hstack((np.eye(3), t_rel)), np.array([0, 0, 0, 1])))
            
            # The extrinsic transformation from Body to camera i is:
            # T_cam_body = T_pi_p0 * T_cam0_body
            T_body_cam = T_p0_pi @ T_body_cam0
            T_cam_body = np.linalg.inv(T_body_cam)
            T_cam_body = T_cam_body[:3, :]
            camera_data[key] = {
                "K": K,
                "T_cam_body": T_cam_body
            }
    
    return camera_data

# def label_sequence(sequence_path, stereo_model):
#     # read camera parameters
#     calib_path = os.path.join(sequence_path, 'calib.txt')
#     if not os.path.exists(calib_path):
#         raise FileNotFoundError(f"Calibration file not found at {calib_path}")
#     parames = read_calibration_file(calib_path)
#     camera_extrinsics = compute_camera_extrinsics(parames, data_type="KRt")

#     camera_2 = camera_extrinsics['P2']['T_cam_body']
#     camera_3 = camera_extrinsics['P3']['T_cam_body']
#     camera_2_k = camera_extrinsics['P2']['K']
#     focal_length = camera_2_k[0, 0]  # Assuming fx is the focal length in pixels

#     baseline = np.linalg.norm(camera_2[:3, 3] - camera_3[:3, 3])
#     print(f"Baseline between camera P2 and P3: {baseline:.2f} meters")
#     left_image_path = os.path.join(sequence_path, 'image_2')
#     right_image_path = os.path.join(sequence_path, 'image_3')
#     depth_npy_path = os.path.join(sequence_path, 'depth_npy')
#     if not os.path.exists(depth_npy_path):
#         os.makedirs(depth_npy_path)

#     image_list = sorted([f for f in os.listdir(left_image_path) if f.endswith('.png')])

#     for image_name in tqdm.tqdm(image_list, desc=f"Processing Seq {os.path.basename(sequence_path)}", unit="image"):

#         if not image_name.endswith('.png'):
#             tqdm.tqdm.write(f"Skipping non-image file: {image_name}")
#             tqdm.tqdm.update(1)
#             tqdm.tqdm.write("Continuing to next image...")
#             continue
#         left_image_file = os.path.join(left_image_path, image_name)
#         right_image_file = os.path.join(right_image_path, image_name)
#         try:
#             # Read the left and right images
#             left_image = np.array(Image.open(left_image_file))
#             left_image_ori = left_image.copy()  # Keep original for visualization
#             right_image = np.array(Image.open(right_image_file))
#             left_image = torch.from_numpy(left_image).permute(2, 0, 1).unsqueeze(0).float().cpu()  # Convert to tensor and move to GPU
#             right_image = torch.from_numpy(right_image).permute(2, 0, 1).unsqueeze(0).float().cpu()
#             padder = InputPadder(left_image.shape, divis_by=32, force_square=False)
#             left_image, right_image = padder.pad(left_image, right_image)
#             left_image = left_image.cuda()  # Move to GPU
#             right_image = right_image.cuda()  # Move to GPU
#             with torch.no_grad(), torch.cuda.amp.autocast(True):
#                 disparity =  stereo_model.forward(left_image, right_image,iters=32,test_mode=True)
#                 disparity = padder.unpad(disparity.float())
#             disparity = disparity.data.cpu().numpy().squeeze(0).squeeze(0) # Move to CPU
#             # confidence = padder.unpad(confidence.float())
#             # confidence = confidence.data.cpu().numpy().squeeze(0).squeeze(0)
#             depth = focal_length * baseline / (disparity + 1e-6)  # Calculate depth in meters
#             np.save(os.path.join(depth_npy_path, image_name.replace('.png', '.npy')), depth)
#         except Exception as e:
#             print(f"Error processing {image_name}: {e}")

        # tqdm.tqdm.write(f"Processed {image_name} - Depth shape: {depth.shape}, Disparity shape: {disparity.shape}, Confidence shape: {confidence.shape}")

        # # 2. 使用Sobel算子计算深度梯度
        # sobel_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # # 计算梯度幅值
        # gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # # 3. 创建高梯度区域的掩码
        # # 此阈值非常关键，可能需要根据效果进行调整。它代表每个像素的深度变化率。
        # gradient_threshold = 2.0 
        # edge_mask = (gradient_magnitude > gradient_threshold).astype(np.uint8)

        # # 4. 膨胀掩码以腐蚀边缘区域
        # # 膨胀操作会扩大掩码区域，从而移除边缘附近的"拉丝"像素。
        # # 可以调整 kernel_size 来控制腐蚀的强度。
        # erosion_kernel_size = 1
        # kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        # dilated_edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

        # # 5. 将掩码应用到深度图
        # filtered_depth_map = depth.copy()
        # # 将被掩码标记的区域（高梯度边缘）的深度值设为0
        # filtered_depth_map[dilated_edge_mask == 1] = 0

        # xyz_map = depth2xyzmap(filtered_depth_map, camera_2_k)  # Convert depth to point cloud
        # pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), left_image_ori.reshape(-1, 3))
        # keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= 128)

        # keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        # pcd = pcd.select_by_index(keep_ids)
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)
        # vis.run()
        # vis.destroy_window()
        # break

def label_sequence(sequence_path, stereo_model, error_log_file):
    # read camera parameters
    calib_path = os.path.join(sequence_path, 'calib.txt')
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found at {calib_path}")
    parames = read_calibration_file(calib_path)
    camera_extrinsics = compute_camera_extrinsics(parames, data_type="KRt")

    camera_2 = camera_extrinsics['P2']['T_cam_body']
    camera_3 = camera_extrinsics['P3']['T_cam_body']
    camera_2_k = camera_extrinsics['P2']['K']
    focal_length = camera_2_k[0, 0]  # Assuming fx is the focal length in pixels

    baseline = np.linalg.norm(camera_2[:3, 3] - camera_3[:3, 3])
    print(f"Baseline between camera P2 and P3: {baseline:.2f} meters")
    left_image_path = os.path.join(sequence_path, 'image_2')
    right_image_path = os.path.join(sequence_path, 'image_3')
    depth_npy_path = os.path.join(sequence_path, 'depth_npy')
    if not os.path.exists(depth_npy_path):
        os.makedirs(depth_npy_path)

    image_list = sorted([f for f in os.listdir(left_image_path) if f.endswith('.png')])

    for image_name in tqdm.tqdm(image_list, desc=f"Processing Seq {os.path.basename(sequence_path)}", unit="image"):

        if not image_name.endswith('.png'):
            tqdm.tqdm.write(f"Skipping non-image file: {image_name}")
            continue
        left_image_file = os.path.join(left_image_path, image_name)
        right_image_file = os.path.join(right_image_path, image_name)
        try:
            # Read the left and right images
            left_image = np.array(Image.open(left_image_file))
            left_image_ori = left_image.copy()  # Keep original for visualization
            right_image = np.array(Image.open(right_image_file))
            left_image = torch.from_numpy(left_image).permute(2, 0, 1).unsqueeze(0).float().cpu()  # Convert to tensor and move to GPU
            right_image = torch.from_numpy(right_image).permute(2, 0, 1).unsqueeze(0).float().cpu()
            padder = InputPadder(left_image.shape, divis_by=32, force_square=False)
            left_image, right_image = padder.pad(left_image, right_image)
            left_image = left_image.cuda()  # Move to GPU
            right_image = right_image.cuda()  # Move to GPU
            with torch.no_grad(), torch.cuda.amp.autocast(True):
                disparity =  stereo_model.forward(left_image, right_image,iters=32,test_mode=True)
                disparity = padder.unpad(disparity.float())
            disparity = disparity.data.cpu().numpy().squeeze(0).squeeze(0) # Move to CPU
            # confidence = padder.unpad(confidence.float())
            # confidence = confidence.data.cpu().numpy().squeeze(0).squeeze(0)
            depth = focal_length * baseline / (disparity + 1e-6)  # Calculate depth in meters
            np.save(os.path.join(depth_npy_path, image_name.replace('.png', '.npy')), depth)
        except Exception as e:
            error_msg = f"Error processing {left_image_file}: {e}\n"
            tqdm.tqdm.write(error_msg)
            with open(error_log_file, 'a') as f:
                f.write(error_msg)
            continue


                
if __name__ == "__main__":
    # Initialize the stereo model (this is a placeholder, replace with actual model initialization)
    cfg = OmegaConf.load('/vepfs-moore-one/peize/Occ-tools/FoundationStereo/pretrained_models/23-51-11/cfg.yaml')  # Load your model configuration
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    ckpt = torch.load('/vepfs-moore-one/peize/Occ-tools/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth')  # Load your model checkpoint
    args = OmegaConf.create(cfg)  # Create an argument object from the configuration and checkpoint
    stereo_model = FoundationStereo(args)  # Replace with actual model initialization
    stereo_model.load_state_dict(ckpt['model'])  # Load the model state dictionary
    
    stereo_model.cuda()  # Move the model to GPU
    stereo_model.eval()  # Set the model to evaluation mode
    print("Model loaded and ready for inference.\n")
    kitti_sequence_root = "/vepfs-moore-one/peize/Occ-dataset/kitti/dataset/sequences/"
    error_log_file = os.path.join(os.path.dirname(kitti_sequence_root.rstrip('/')), "error.log")

    # 清空旧的错误日志
    if os.path.exists(error_log_file):
        os.remove(error_log_file)

    for seq in os.listdir(kitti_sequence_root):
        sequence_path = os.path.join(kitti_sequence_root, seq)
        if os.path.isfile(sequence_path):
            continue
        try:
            label_sequence(sequence_path, stereo_model, error_log_file)  # Call the function to label the sequence
        except Exception as e:
            error_msg = f"Failed to process sequence {sequence_path}: {e}\n"
            print(error_msg)
            with open(error_log_file, 'a') as f:
                f.write(error_msg)
    
    print("\nAll sequences processed.")
    if os.path.exists(error_log_file) and os.path.getsize(error_log_file) > 0:
        print(f"Some files failed to process. Check '{error_log_file}' for details.")
    else:
        print("No errors were logged.")