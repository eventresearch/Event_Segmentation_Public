import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as Rot
import hdf5plugin
import os
from .eventreader import EventReader

def load_yaml_file(file_path):
    """
    Load a YAML file and return its contents as a Python dictionary.
    Args:
        file_path (str): Path to the YAML file.
    Returns:
        dict: Parsed YAML data.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def _to_K_3x3(k_4):
    """
    Convert DSEC-style camera intrinsics [fx, fy, cx, cy] to a 3x3 camera matrix K.
    Args:
        k_4 (list or array): [fx, fy, cx, cy]
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix.
    """
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = float(k_4[0])  # Focal length x
    K[1, 1] = float(k_4[1])  # Focal length y
    K[0, 2] = float(k_4[2])  # Principal point x
    K[1, 2] = float(k_4[3])  # Principal point y
    return K


def rectification_params_from_yaml_event(calib_file, cam_number: int = 0):
    """
    Reads calibration parameters from a YAML file for RGB or event cameras.
    Args:
        calib_file (str): Path to calibration YAML file.
        cam_number (int): 0 for left event camera, 1 for left RGB camera.
    Returns:
        tuple: (distMat, dist_coeffs, R_rect, rectMat, resolution, cam_number)
            distMat (np.ndarray): Distorted camera matrix (3x3)
            dist_coeffs (np.ndarray): Distortion coefficients
            R_rect (np.ndarray): Rectification rotation matrix
            rectMat (np.ndarray): Rectified camera matrix (3x3)
            resolution (tuple): (width, height) of rectified image
            cam_number (int): Camera number used
    """
    if cam_number != 0 and cam_number != 1:
        raise ValueError("cam_number must be 0 or 1")
    if cam_number == 0:
        print("Using LEFT event camera parameters (cam0 → camRect0)")
    else:
        print("Using LEFT RGB camera parameters (cam1 → camRect1)")

    # cam_number=0 → cam0/camRect0; cam_number=1 → cam1/camRect1
    cam = f"cam{cam_number}"
    camR = f"camRect{cam_number}"
    R_key = f"R_rect{cam_number}"

    yaml_data = load_yaml_file(calib_file)

    # Extract camera intrinsics and extrinsics
    K_dist = yaml_data['intrinsics'][cam]['camera_matrix']        # [fx, fy, cx, cy]
    dist_coeffs = yaml_data['intrinsics'][cam]['distortion_coeffs']
    R_rect = np.array(yaml_data['extrinsics'][R_key], dtype=np.float32)
    K_rect = yaml_data['intrinsics'][camR]['camera_matrix']    # [fx, fy, cx, cy]
    resolution = yaml_data['intrinsics'][camR]['resolution']   # [W, H]

    # Build 3x3 intrinsics
    distMat = _to_K_3x3(K_dist)
    rectMat = _to_K_3x3(K_rect)

    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)
    resolution = (int(resolution[0]), int(resolution[1]))  # (W, H)

    return distMat, dist_coeffs, R_rect, rectMat, resolution, cam_number

def rectification_params_from_yaml_rgb(calib_file, high_res=True, isotropic: bool = True, return_map: bool = False, border_value: int = 0, border_mode=cv2.BORDER_CONSTANT):
    """
    Reads rectification parameters for RGB cameras from a YAML file.
    Args:
        calib_file (str): Path to calibration YAML file.
        high_res (bool): If True, scale intrinsics for high-res output.
        isotropic (bool): If True, scale x and y equally.
        return_map (bool): If True, return pixel mapping for fast remap.
        border_value (int): Value for constant border (default: 0).
        border_mode: OpenCV border mode (default: cv2.BORDER_CONSTANT).
    Returns:
        tuple: (P_r1_from_r0, (W0, H0), (W1, H1), interp/flags, border_mode, border_value, maps)
            P_r1_from_r0 (np.ndarray): Homography matrix for warpPerspective
            (W0, H0): Output resolution (Rect0)
            (W1, H1): Input resolution (Rect1)
            interp/flags: Interpolation flags for OpenCV (warpPerspective or remap)
            border_mode: Border mode for OpenCV
            border_value: Value for border
            maps: (map1_fp16, map2_fp32) for fast remap if return_map=True, else None
    """
    yaml_data = load_yaml_file(calib_file)

    # 1) Intrinsics
    K_rect0 = _to_K_3x3(yaml_data['intrinsics']['camRect0']['camera_matrix']).astype(np.float32)
    K_rect1 = _to_K_3x3(yaml_data['intrinsics']['camRect1']['camera_matrix']).astype(np.float32)

    # 2) Resolutions
    W0, H0 = map(int, yaml_data['intrinsics']['camRect0']['resolution'])
    W1, H1 = map(int, yaml_data['intrinsics']['camRect1']['resolution'])

    # 3) Rectifying rotations (float32)
    R_r0_0 = Rot.from_matrix(np.array(yaml_data['extrinsics']['R_rect0'], dtype=np.float32))
    R_r1_1 = Rot.from_matrix(np.array(yaml_data['extrinsics']['R_rect1'], dtype=np.float32))

    # 4) Raw-frame transform cam1->cam0 (use rotation only)
    T_10 = np.array(yaml_data['extrinsics']['T_10'], dtype=np.float32)
    R_10 = Rot.from_matrix(T_10[:3, :3])

    # 5) Rotation: camRect0 -> camRect1
    R_r1_r0 = (R_r1_1 * R_10 * R_r0_0.inv()).as_matrix().astype(np.float32)

    K0_for_warp = None

    interp = None
    # 6) Possibly scale Rect0 intrinsics (super-Rect0 canvas)
    if high_res is True:
        # Scale Rect0 intrinsics to a high-res Rect0 canvas
        sx = W1 / W0
        sy = H1 / H0
        if isotropic:
            s = min(sx, sy)
            sx = sy = s

        K_0_p = K_rect0.copy()
        K_0_p[0, 0] *= sx  # fx'
        K_0_p[1, 1] *= sy  # fy'
        K_0_p[0, 2] *= sx  # cx'
        K_0_p[1, 2] *= sy  # cy'

        W0 = int(round(W0 * sx))
        H0 = int(round(H0 * sy))

        K0_for_warp = K_0_p
        # For high-res, use linear interpolation to better preserve details
        interp = cv2.INTER_LINEAR

    elif high_res is False:
        K0_for_warp = K_rect0
        # Use area interpolation for downsampling
        interp = cv2.INTER_AREA
    else:
        raise ValueError("high_res must be True or False")

    # 7) Homography for warpPerspective (dst = Rect0 canvas, src = Rect1)
    P_r1_from_r0 = (K_rect1 @ R_r1_r0 @ np.linalg.inv(K0_for_warp)).astype(np.float32)

    if return_map:
        # Build dst grid (camRect0 canvas) and project to src coords (camRect1)
        xs, ys = np.meshgrid(np.arange(W0, dtype=np.float32),
                             np.arange(H0, dtype=np.float32))
        ones = np.ones_like(xs)
        grid = np.stack([xs, ys, ones], axis=-1)         # (H0,W0,3)
        src = grid @ P_r1_from_r0.T                      # (H0,W0,3)
        src = src[..., :2] / src[..., 2:3]               # perspective divide
        maps = src.astype(np.float32)
        # Fast remap maps
        map1_fp16, map2_fp32 = cv2.convertMaps(maps, None, cv2.CV_16SC2)
        return P_r1_from_r0, (W0, H0), (W1, H1), interp, border_mode, border_value, (map1_fp16, map2_fp32)
    else:
        # Add inverse map flag for warpPerspective
        flags = interp | cv2.WARP_INVERSE_MAP
        return P_r1_from_r0, (W0, H0), (W1, H1), flags, border_mode, border_value, None

def rectify_rgb_frame(image_rect1, calib_params):
    """
    Rectify an RGB image using warpPerspective and calibration parameters.
    Args:
        image_rect1 (np.ndarray): Input image (Rect1 canvas)
        calib_params (tuple): Calibration parameters from rectification_params_from_yaml_rgb
    Returns:
        np.ndarray: Rectified image in Rect0 canvas
    """
    P_r1_from_r0, (W0, H0), (W1, H1), flags, border_mode, border_value, _ = calib_params

    if image_rect1.shape[:2] != (H1, W1):
        raise ValueError(f"expected {(H1, W1)}, got {image_rect1.shape[:2]}")

    # Output in camRect0 canvas
    rgb_in_rect0 = cv2.warpPerspective(
        image_rect1, P_r1_from_r0, dsize=(W0, H0),
        flags=flags, borderMode=border_mode, borderValue=border_value
    )
    return rgb_in_rect0

def rectify_rgb_frame_fast(
    image_rect1, calib_params
):
    """
    Fast rectification of an RGB image using precomputed pixel mapping.
    Args:
        image_rect1 (np.ndarray): Input image (Rect1 canvas)
        calib_params (tuple): Calibration parameters with pixel map, as returned by rectification_params_from_yaml_rgb (with return_map=True)
            Should include: interp, border_mode, border_value, (map1_fp16, map2_fp32)
    Returns:
        np.ndarray: Rectified image in Rect0 canvas
    """
    P_r1_from_r0, (W0, H0), (W1, H1), interp, border_mode, border_value, maps = calib_params

    if maps is None:
        raise ValueError("map is None, cannot use fast remap method")
    if image_rect1.shape[:2] != (H1, W1):
        raise ValueError(f"expected {(H1, W1)}, got {image_rect1.shape[:2]}")
    map1_fp16, map2_fp32 = maps
    # Output in camRect0 canvas
    rgb_in_rect0 = cv2.remap(
        image_rect1, map1_fp16, map2_fp32,
        interpolation=interp,
        borderMode=border_mode,
        borderValue=border_value
    )
    return rgb_in_rect0


def rectify_event_frame(image, calib_params):
    """
    Rectify an image using undistort and rectify mapping from calibration parameters.
    Args:
        image (np.ndarray): Input image
        calib_params (tuple): Calibration parameters from rectification_params_from_yaml_event
    Returns:
        np.ndarray: Rectified image
    """
    distMat, dist_coeffs, R_rect, rectMat, resolution, cam_number = calib_params

    # Choose interpolation type based on camera
    interpolation_type = cv2.INTER_NEAREST if cam_number == 0 else cv2.INTER_CUBIC

    # Obtain the rectifying mapping from camera variables
    mapping = cv2.initUndistortRectifyMap(
        cameraMatrix=distMat,
        distCoeffs=dist_coeffs,
        R=R_rect,
        newCameraMatrix=rectMat,
        size=resolution,
        m1type=cv2.CV_32FC2
    )[0]  # Get the mappings as 2 channel float32 array and discard the second output

    if image is None:
        print(f"Skipping invalid image")
        return None

    # Rectify the image
    rect_representation = cv2.remap(
        src=image,
        map1=mapping,
        map2=None,
        interpolation=interpolation_type
    )
    return rect_representation


def calculate_pixel_lengths(frame_shape, max_length, min_length=None):
    """
    Calculate a length value for each pixel in a frame, where the center pixel has the maximum length
    and the border pixels have the minimum length (default: max_length/10).
    Args:
        frame_shape (tuple): (height, width) of the frame
        max_length (float): Maximum length at the center
        min_length (float, optional): Minimum length at the borders
    Returns:
        np.ndarray: 2D array of pixel lengths
    """
    height, width = frame_shape
    if min_length is None:
        min_length = max_length / 10

    # Create coordinate grids for pixel locations
    y = np.arange(height)
    x = np.arange(width)
    X, Y = np.meshgrid(x, y)

    # Find center coordinates
    center_x = width / 2
    center_y = height / 2

    # Calculate distance from center for each pixel
    max_distance = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    normalized_distance = distance / max_distance

    # Linear interpolation: length = max at center (dist=0), min at corners (dist=1)
    pixel_lengths = max_length - (max_length - min_length) * normalized_distance
    return pixel_lengths.astype(np.uint32)

def calculate_pixel_lengths_power(frame_shape, max_length, min_length=None, gamma=2.0):
    """
    Calculate a length value for each pixel in a frame using a power-law falloff.
    The center pixel has the maximum length, and values decrease toward the borders
    according to (1 - r)^gamma instead of a linear interpolation.

    Args:
        frame_shape (tuple): (height, width) of the frame.
        max_length (float): Maximum length at the center.
        min_length (float, optional): Minimum length at the borders.
        gamma (float): Power exponent controlling how quickly lengths drop off.
                       Higher gamma => stronger suppression near borders.

    Returns:
        np.ndarray: 2D array of pixel lengths (uint32).
    """
    height, width = frame_shape
    if min_length is None:
        min_length = max_length / 10.0

    y = np.arange(height)
    x = np.arange(width)
    X, Y = np.meshgrid(x, y)

    center_x = width / 2.0
    center_y = height / 2.0

    max_distance = np.sqrt((width / 2.0)**2 + (height / 2.0)**2)
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    normalized_distance = distance / max_distance

    # Power-law weight
    # w = (1 - r)^gamma, r in [0, 1]
    w = (1.0 - normalized_distance)**gamma
    w = np.clip(w, 0.0, 1.0)

    # Map to [min_length, max_length]
    pixel_lengths = min_length + (max_length - min_length) * w

    return pixel_lengths.astype(np.uint32)


def calculate_pixel_lengths_rational(frame_shape, max_length, min_length=None, gamma=2.0, r0=0.45):
    """
    Calculate a length value for each pixel in a frame using a rational / power falloff.
    
    dt(r) = dt_min + (dt_max - dt_min) / (1 + (r / r0)^gamma)
    
    Args:
        frame_shape (tuple): (height, width) of the frame.
        max_length (float): Maximum length at the center (dt_max).
        min_length (float, optional): Minimum length at the borders (dt_min).
        r0 (float): Normalized radius where the drop starts ("knee").
        gamma (float): Power exponent controlling how sharp the drop is.
        
    Returns:
        np.ndarray: 2D array of pixel lengths (uint32).
    """
    height, width = frame_shape
    if min_length is None:
        min_length = max_length / 10.0

    y = np.arange(height)
    x = np.arange(width)
    X, Y = np.meshgrid(x, y)

    center_x = width / 2.0
    center_y = height / 2.0

    # Calculate normalized distance r in [0, 1]
    # We use the max possible distance (corner to center) for normalization to keep r in [0, ~1]
    # or should we normalize by min(width, height)/2 ?
    # The prompt says: "Define normalized radius r in [0, 1] from a center"
    # Usually strictly [0,1] implies usually normalizing by half-diagonal or similar maximum extent.
    max_distance = np.sqrt((width / 2.0)**2 + (height / 2.0)**2)
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    r = distance / max_distance
    
    # dt(r) = dt_min + (dt_max - dt_min) / (1 + (r / r0)^gamma)
    pixel_lengths = min_length + (max_length - min_length) / (1 + (r / r0)**gamma)

    return pixel_lengths.astype(np.uint32)


def render_events(
    x: np.ndarray,
    y: np.ndarray,
    pol: np.ndarray,
    t: np.ndarray,
    frame_shape: tuple[int, int],
    time_lookup: np.ndarray = None
) -> np.ndarray:
    """
    Render an event frame from event data arrays.
    Args:
        x (np.ndarray): x-coordinates of events
        y (np.ndarray): y-coordinates of events
        pol (np.ndarray): polarity of events (1 or 0)
        t (np.ndarray): timestamps of events
        frame_shape (tuple): (height, width) of output frame
        time_lookup (np.ndarray, optional): Per-pixel time threshold for filtering events
    Returns:
        np.ndarray: RGB image visualizing events
            - Red for negative polarity
            - Blue for positive polarity
            - White for no event
    """
    assert x.size == y.size == pol.size
    FrameHeight, FrameWidth = frame_shape
    assert FrameHeight > 0
    assert FrameWidth > 0

    # Initialize output image (white background) and mask
    img = np.full((FrameHeight, FrameWidth, 3), fill_value=255, dtype='uint8')
    mask = np.zeros((FrameHeight, FrameWidth), dtype='int32')

    # Convert polarity: 0 -> -1 (negative), 1 -> 1 (positive)
    pol = pol.astype('int')
    pol[pol == 0] = -1

    # Round coordinates to nearest integer pixel
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    # Filter out-of-bounds coordinates
    valid_coords = (x >= 0) & (x < FrameWidth) & (y >= 0) & (y < FrameHeight)

    if time_lookup is not None:
        # Get the time threshold for each event based on its pixel location
        time_threshold = time_lookup[y[valid_coords], x[valid_coords]]
        # Keep only events that occur at or after their pixel's threshold
        time_events = t[valid_coords] - np.min(t[valid_coords])
        max_time = np.max(time_events)
        keep_mask_relative = max_time - time_events < time_threshold
        # Get the indices where valid_coords is True
        valid_indices = np.where(valid_coords)[0]
        # Among those valid indices, keep only the ones that pass the time threshold
        final_indices = valid_indices[keep_mask_relative]
    else:
        final_indices = valid_coords

    # Use final_indices to render only the filtered events if time_lookup is provided, else use all valid events
    mask[y[final_indices], x[final_indices]] = pol[final_indices]

    # Color code: 0 = white, -1 = red (negative), 1 = blue (positive)
    img[mask == 0] = [255, 255, 255]
    img[mask == -1] = [255, 0, 0]
    img[mask == 1] = [0, 0, 255]
    return img

def crop_images_to_640x440(directory):
    """
    Crop all 640x480 images in the specified directory to 640x440 by removing the bottom part.
    Overwrites the original files.
    """
    # Convert directory to Path object
    directory_path = Path(directory)

    # Check if directory exists
    if not directory_path.is_dir():
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    # Process each image file in the directory
    for image_file in tqdm(directory_path.glob("*.png"), desc="Cropping images"):
        # Read the image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Skipping invalid image: {image_file}")
            continue

        # Check if the image size is 640x480
        if image.shape[:2] != (480, 640):
            print(f"Skipping non-640x480 image: {image_file}")
            continue

        # Crop the image to 640x440
        cropped_image = image[:440, :]

        # Overwrite the file with the cropped image
        cv2.imwrite(str(image_file), cropped_image)

    print("Cropping complete.")


os.environ['HDF5_PLUGIN_PATH'] = os.path.dirname(hdf5plugin.__file__)

def event_to_frame(event_filepath, event_delta_time_ms=50, high_low_ratio=10, rgb_delta_time_ms=50, use_time_lookup=True, use_calibration=True, output_path=None, calibration_file=None, frame_shape=(480, 640), gamma=2.0, r0=None):
    # Validate timing parameters
    if event_delta_time_ms <= 0:
        raise ValueError("event_delta_time_ms must be > 0")
    if rgb_delta_time_ms <= 0:
        raise ValueError("rgb_delta_time_ms must be > 0")

    # Calculate how many event frames to skip. Ensure non-negative so (frames_to_skip+1) > 0
    frames_to_skip = int(rgb_delta_time_ms / event_delta_time_ms) - 1
    if frames_to_skip < 0:
        frames_to_skip = 0
    
    # for 50 ms fps is 20 so we need to adjust for add dt time to have same lenght video, the rgb delta time is 50ms. Since for longer event accumulation time we would have less frames we need to have lower fps to have same length video
    fps = 1000.0 / rgb_delta_time_ms
    
    if output_path is None:
        raise ValueError("output_path must be provided")
    output_path = Path(output_path)

    is_video = False
    writer = None
    if output_path.suffix.lower() in ['.mp4', '.avi', '.mkv']:
        is_video = True

    output_dir = output_path if not is_video else output_path.parent
    
    os.makedirs(output_dir, exist_ok=True)
    
    rectification_params_event = None
    if use_calibration:
        if calibration_file is None:
            raise ValueError("Calibration file must be provided if use_calibration is True.")
        rectification_params_event = rectification_params_from_yaml_event(calibration_file, cam_number=0)

    time_lookup = None
    if use_time_lookup:
        # time_lookup = calculate_pixel_lengths(frame_shape, max_length=event_delta_time_ms*1000, min_length=(event_delta_time_ms*1000)/high_low_ratio)
        if r0 is not None:
            time_lookup = calculate_pixel_lengths_rational(frame_shape, max_length=event_delta_time_ms*1000, min_length=(event_delta_time_ms*1000)/high_low_ratio, gamma=gamma, r0=r0)
        else:
            time_lookup = calculate_pixel_lengths_power(frame_shape, max_length=event_delta_time_ms*1000, min_length=(event_delta_time_ms*1000)/high_low_ratio, gamma=gamma)



    event_frame_index = 0
    output_frame_index = 0

    if event_delta_time_ms > rgb_delta_time_ms:
        # Buffered Mode: Overlapping windows
        # Read events in small chunks (rgb_delta_time_ms) and accumulate in a buffer
        reader_dt = rgb_delta_time_ms
        event_delta_time_us = int(event_delta_time_ms * 1000)
        
        # Buffer stores tuples of (events_dict, chunk_end_time_us)
        event_buffer = deque()

        # Initialize EventReader with rgb_delta_time_ms step size
        # We need to access the reader instance to get authoritative time
        reader = EventReader(event_filepath, reader_dt)
        
        # Robust time tracking: use the reader's state which tracks the definitive bin edges
        # EventReader.t_start_us is updated to the NEXT start time (which is current end time)
        # just before yielding in __next__.
        
        for i, events in enumerate(tqdm(reader)):
            # Get the authoritative end time of this chunk from the reader
            current_t_end_us = reader.t_start_us
            
            # Append current chunk to buffer
            event_buffer.append((events, current_t_end_us))
            
            # Prune buffer: remove chunks that end before the window starts
            # Window is [current_t_end_us - event_delta_time_us, current_t_end_us]
            window_start_us = current_t_end_us - event_delta_time_us
            
            while event_buffer and event_buffer[0][1] <= window_start_us:
                event_buffer.popleft()
            
            # Check if we have any events in the buffer
            total_events = 0
            for ev, _ in event_buffer:
                if ev is not None and 't' in ev:
                    total_events += ev['t'].size
            
            if total_events > 0:
                xs, ys, ps, ts = [], [], [], []
                for ev, _ in event_buffer:
                    xs.append(ev['x'])
                    ys.append(ev['y'])
                    ps.append(ev['p'])
                    ts.append(ev['t'])

                x_concat = np.concatenate(xs)
                y_concat = np.concatenate(ys)
                p_concat = np.concatenate(ps)
                t_concat = np.concatenate(ts)
                
                # Precise Masking: Filter events that are strictly within the window
                # The oldest chunk might partially overlap the window start
                mask = (t_concat > window_start_us) & (t_concat <= current_t_end_us)
                
                x = x_concat[mask]
                y = y_concat[mask]
                p = p_concat[mask]
                t = t_concat[mask]
            else:
                # Handle safe empty case with correct dtypes
                x = np.array([], dtype=np.int16)
                y = np.array([], dtype=np.int16)
                p = np.array([], dtype=np.int8)
                t = np.array([], dtype=np.int64)

            # Render
            img = render_events(x, y, p, t, frame_shape, time_lookup)

            if use_calibration:
                img = rectify_event_frame(img, rectification_params_event)

            if is_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(img.shape[1], img.shape[0]))
                    
                writer.write(img)
            else:
                cv2.imwrite(f'{output_dir}/{output_frame_index:06d}.png', img)

            
                
            output_frame_index += 1

        if writer is not None:
            writer.release()

    else:
        # Legacy Mode: Non-overlapping or skipping windows (event_delta <= rgb_delta)
        # Keeps exact original behavior for regression testing safety
        for i, events in enumerate(tqdm(EventReader(event_filepath, event_delta_time_ms))):
            # Check if the current event frame should be processed
            # Only process the last event frame before each RGB frame
            if (event_frame_index + 1) % (frames_to_skip + 1) == 0:
                p = events['p']
                x = events['x']
                y = events['y']
                t = events['t']

                img = render_events(x, y, p, t, frame_shape, time_lookup)

                if use_calibration:
                    img = rectify_event_frame(img, rectification_params_event)

                if is_video:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(img.shape[1], img.shape[0]))
                        
                    writer.write(img)
                else:
                    cv2.imwrite(f'{output_dir}/{output_frame_index:06d}.png', img)
                    
                # Increment output frame index
                output_frame_index += 1
        
            # Increment event frame index
            event_frame_index += 1

        if writer is not None:
            writer.release()
    
def rectify_rgb_images(rgb_folder_path=None, output_path=None, calibration_file=None, high_res=False, isotropic=False, fps=20):
    is_video = False
    writer = None
    if output_path.suffix.lower() in ['.mp4', '.avi', '.mkv']:
        is_video = True

    output_dir = output_path if not is_video else output_path.parent
    
    os.makedirs(output_dir, exist_ok=True)
    
    rectification_params_rgb = None

    rectification_params_rgb = rectification_params_from_yaml_rgb(calibration_file, high_res=high_res, isotropic=isotropic)

    for i, img_file in enumerate(tqdm(sorted(rgb_folder_path.glob("*.png")))):
        img = cv2.imread(str(img_file))
        img = rectify_rgb_frame(img, rectification_params_rgb)

        if is_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(img.shape[1], img.shape[0]))

            writer.write(img)
        else:
            cv2.imwrite(f'{output_dir}/{i:06d}.png', img)