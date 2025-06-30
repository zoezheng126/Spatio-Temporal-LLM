import numpy as np
from scipy.spatial.transform import Rotation as R
from vis import generate_masks_for_image 
import os
import random 
import json 
from colmap_utils import read_model
import pandas as pd
from tqdm import tqdm
import time
import ast
from qwen_utils import VLMPredictor

random.seed(0)
np.random.seed(0)

rgb_main_dir = "EPIC-KITCHENS/frames_rgb_flow/rgb"  # ! NEED TO BE UPDATED
visor_to_epic_map_path = "../annotations/frame_mapping_train_val.json"
colmap_main_dir = "../output"
colmap_model_main_dir = "Epic-Field/datasets/Epic_Field/dense"
visor_gt_json_main_dir = "GroundTruth-SparseAnnotations/annotations"  # ! NEED TO BE UPDATED

small_objects_id_to_name_dict = {1: 'spoon', 2: 'plate', 4: 'knife', 5: 'pan', 6: 'lid', 7: 'bowl', 9: 'sponge', 10: 'glass', 13: 'cup', 14: 'fork', 15: 'bottle', 16: 'onion', 19: 'bag', 20: 'spatula', 21: 'container', 22: 'liquid:washing', 23: 'box', 25: 'dough', 26: 'package', 28: 'meat', 29: 'pot', 30: 'potato', 31: 'oil', 32: 'cheese', 33: 'bread', 35: 'tray', 36: 'bin', 37: 'pepper', 38: 'salt', 39: 'colander', 40: 'jar', 41: 'carrot', 43: 'tomato', 44: 'kettle', 45: 'pasta', 47: 'sauce', 48: 'skin', 49: 'paper', 51: 'garlic', 52: 'towel', 53: 'egg', 55: 'rice', 56: 'mushroom', 57: 'chicken', 58: 'cutlery', 59: 'coffee', 60: 'glove', 61: 'can', 62: 'leaf', 64: 'milk', 66: 'jug', 67: 'aubergine', 68: 'salad', 69: 'chilli', 71: 'mixture', 72: 'cucumber', 73: 'clothes', 74: 'peach', 75: 'flour', 76: 'courgette', 78: 'butter', 79: 'scissors', 80: 'chopstick', 81: 'tofu', 83: 'olive', 84: 'mat', 85: 'spice', 86: 'sausage', 87: 'peeler:potato', 88: 'napkin', 89: 'cover', 91: 'pizza', 92: 'button', 93: 'towel:kitchen', 94: 'vegetable', 95: 'stock', 96: 'grater', 97: 'ladle', 98: 'yoghurt', 99: 'cereal', 100: 'wrap:plastic', 101: 'broccoli', 102: 'sugar', 103: 'brush', 104: 'biscuit', 105: 'lemon', 106: 'juicer', 107: 'wrap', 108: 'scale', 109: 'rest', 111: 'alarm', 112: 'salmon', 114: 'light', 115: 'spreads', 116: 'squash', 117: 'leek', 118: 'cap', 119: 'fish', 120: 'lettuce', 121: 'curry', 122: 'seed', 123: 'foil', 125: 'corn', 126: 'soup', 127: 'oatmeal', 128: 'onion:spring', 129: 'clip', 130: 'lighter', 131: 'ginger', 132: 'tea', 133: 'nut', 134: 'vinegar', 135: 'holder', 136: 'pin:rolling', 137: 'pie', 138: 'powder', 139: 'burger', 140: 'book', 141: 'shell:egg', 142: 'tongs', 143: 'cream', 144: 'pork', 145: 'oregano', 146: 'banana', 148: 'paste', 149: 'recipe', 150: 'liquid', 151: 'choi:pak', 155: 'noodle', 156: 'salami', 158: 'teapot', 161: 'lime', 162: 'omelette', 163: 'bacon', 164: 'sandwich', 165: 'phone', 166: 'thermometer', 167: 'orange', 168: 'basket', 169: 'parsley', 171: 'tablet', 173: 'coriander', 174: 'opener:bottle', 175: 'cake', 176: 'avocado', 177: 'lentil', 178: 'blueberry', 181: 'hummus', 183: 'juice', 184: 'pancake', 185: 'bean:green', 187: 'apple', 188: 'chocolate', 189: 'ice', 190: 'knob', 191: 'handle', 192: 'wine', 193: 'pea', 194: 'pith', 195: 'yeast', 196: 'coconut', 197: 'fishcakes', 198: 'spinach', 199: 'apron', 200: 'raisin', 201: 'basil', 202: 'grape', 203: 'kale', 205: 'asparagus', 206: 'paprika', 207: 'mango', 208: 'caper', 209: 'drink', 210: 'stalk', 211: 'turmeric', 212: 'whetstone', 213: 'kiwi', 214: 'bean', 215: 'thyme', 216: 'finger:lady', 217: 'beef', 218: 'whisk', 219: 'blackberry', 220: 'slicer', 222: 'label', 223: 'celery', 224: 'cabbage', 226: 'breadstick', 227: 'roll', 228: 'cocktail', 229: 'crisp', 231: 'beer', 233: 'battery', 234: 'powder:washing', 235: 'backpack', 236: 'cumin', 237: 'cutter:pizza', 238: 'air', 239: 'pear', 240: 'quorn', 241: 'funnel', 243: 'strawberry', 244: 'almond', 246: 'scotch:egg', 248: 'straw', 251: 'masher', 252: 'guard:hand', 253: 'shrimp', 254: 'fruit', 255: 'artichoke', 256: 'cork', 257: 'cherry', 258: 'sprout', 259: 'mat:sushi', 260: 'stick:crab', 261: 'ring:onion', 262: 'pestle', 264: 'gin', 265: 'bar', 266: 'mint', 268: 'grass:lemon', 269: 'rubber', 270: 'gherkin', 271: 'breadcrumb', 272: 'watch', 273: 'melon', 274: 'cinnamon', 275: 'popcorn', 276: 'dumpling', 277: 'rosemary', 279: 'syrup', 280: 'candle', 281: 'pineapple', 282: 'sheets', 283: 'soda', 284: 'raspberry', 286: 'balloon', 287: 'turkey', 289: 'key', 290: 'pillow', 291: 'pen', 293: 'plum', 294: 'whiskey', 296: 'tape', 297: 'camera', 298: 'cd', 299: 'extract:vanilla'}
small_objects_name_to_id_dict = {'spoon': 1, 'plate': 2, 'knife': 4, 'pan': 5, 'lid': 6, 'bowl': 7, 'sponge': 9, 'glass': 10, 'cup': 13, 'fork': 14, 'bottle': 15, 'onion': 16, 'bag': 19, 'spatula': 20, 'container': 21, 'liquid:washing': 22, 'box': 23, 'dough': 25, 'package': 26, 'meat': 28, 'pot': 29, 'potato': 30, 'oil': 31, 'cheese': 32, 'bread': 33, 'tray': 35, 'bin': 36, 'pepper': 37, 'salt': 38, 'colander': 39, 'jar': 40, 'carrot': 41, 'tomato': 43, 'kettle': 44, 'pasta': 45, 'sauce': 47, 'skin': 48, 'paper': 49, 'garlic': 51, 'towel': 52, 'egg': 53, 'rice': 55, 'mushroom': 56, 'chicken': 57, 'cutlery': 58, 'coffee': 59, 'glove': 60, 'can': 61, 'leaf': 62, 'milk': 64, 'jug': 66, 'aubergine': 67, 'salad': 68, 'chilli': 69, 'mixture': 71, 'cucumber': 72, 'clothes': 73, 'peach': 74, 'flour': 75, 'courgette': 76, 'butter': 78, 'scissors': 79, 'chopstick': 80, 'tofu': 81, 'olive': 83, 'mat': 84, 'spice': 85, 'sausage': 86, 'peeler:potato': 87, 'napkin': 88, 'cover': 89, 'pizza': 91, 'button': 92, 'towel:kitchen': 93, 'vegetable': 94, 'stock': 95, 'grater': 96, 'ladle': 97, 'yoghurt': 98, 'cereal': 99, 'wrap:plastic': 100, 'broccoli': 101, 'sugar': 102, 'brush': 103, 'biscuit': 104, 'lemon': 105, 'juicer': 106, 'wrap': 107, 'scale': 108, 'rest': 109, 'alarm': 111, 'salmon': 112, 'light': 114, 'spreads': 115, 'squash': 116, 'leek': 117, 'cap': 118, 'fish': 119, 'lettuce': 120, 'curry': 121, 'seed': 122, 'foil': 123, 'corn': 125, 'soup': 126, 'oatmeal': 127, 'onion:spring': 128, 'clip': 129, 'lighter': 130, 'ginger': 131, 'tea': 132, 'nut': 133, 'vinegar': 134, 'holder': 135, 'pin:rolling': 136, 'pie': 137, 'powder': 138, 'burger': 139, 'book': 140, 'shell:egg': 141, 'tongs': 142, 'cream': 143, 'pork': 144, 'oregano': 145, 'banana': 146, 'paste': 148, 'recipe': 149, 'liquid': 150, 'choi:pak': 151, 'noodle': 155, 'salami': 156, 'teapot': 158, 'lime': 161, 'omelette': 162, 'bacon': 163, 'sandwich': 164, 'phone': 165, 'thermometer': 166, 'orange': 167, 'basket': 168, 'parsley': 169, 'tablet': 171, 'coriander': 173, 'opener:bottle': 174, 'cake': 175, 'avocado': 176, 'lentil': 177, 'blueberry': 178, 'hummus': 181, 'juice': 183, 'pancake': 184, 'bean:green': 185, 'apple': 187, 'chocolate': 188, 'ice': 189, 'knob': 190, 'handle': 191, 'wine': 192, 'pea': 193, 'pith': 194, 'yeast': 195, 'coconut': 196, 'fishcakes': 197, 'spinach': 198, 'apron': 199, 'raisin': 200, 'basil': 201, 'grape': 202, 'kale': 203, 'asparagus': 205, 'paprika': 206, 'mango': 207, 'caper': 208, 'drink': 209, 'stalk': 210, 'turmeric': 211, 'whetstone': 212, 'kiwi': 213, 'bean': 214, 'thyme': 215, 'finger:lady': 216, 'beef': 217, 'whisk': 218, 'blackberry': 219, 'slicer': 220, 'label': 222, 'celery': 223, 'cabbage': 224, 'breadstick': 226, 'roll': 227, 'cocktail': 228, 'crisp': 229, 'beer': 231, 'battery': 233, 'powder:washing': 234, 'backpack': 235, 'cumin': 236, 'cutter:pizza': 237, 'air': 238, 'pear': 239, 'quorn': 240, 'funnel': 241, 'strawberry': 243, 'almond': 244, 'scotch:egg': 246, 'straw': 248, 'masher': 251, 'guard:hand': 252, 'shrimp': 253, 'fruit': 254, 'artichoke': 255, 'cork': 256, 'cherry': 257, 'sprout': 258, 'mat:sushi': 259, 'stick:crab': 260, 'ring:onion': 261, 'pestle': 262, 'gin': 264, 'bar': 265, 'mint': 266, 'grass:lemon': 268, 'rubber': 269, 'gherkin': 270, 'breadcrumb': 271, 'watch': 272, 'melon': 273, 'cinnamon': 274, 'popcorn': 275, 'dumpling': 276, 'rosemary': 277, 'syrup': 279, 'candle': 280, 'pineapple': 281, 'sheets': 282, 'soda': 283, 'raspberry': 284, 'balloon': 286, 'turkey': 287, 'key': 289, 'pillow': 290, 'pen': 291, 'plum': 293, 'whiskey': 294, 'tape': 296, 'camera': 297, 'cd': 298, 'extract:vanilla': 299}

anchor_objects_id_to_name_dict = {
    12: "fridge",
    24: "hob",
    46: "oven",
    63: "sink",
    70: "dishwasher",
    113: "freezer",
    124: "machine:washing",
    245: "tv",
    90: "microwave",
    50: "maker:coffee",
}
anchor_objects_name_to_id_dict = {
    "fridge": 12,
    "hob": 24,
    "oven": 46,
    "sink": 63,
    "dishwasher": 70,
    "freezer": 113,
    "machine:washing": 124,
    "tv": 245,
    "microwave": 90,
    "maker:coffee": 50,
}

def relative_direction(qvec1, tvec1, qvec2, tvec2, trans_threshold=1.5):
    """
    Calculate the relative movement direction between two poses. from pose1 to pose2.

    Parameters
    ----------
    qvec1, qvec2 : array-like of shape (4,)
        Quaternions (qw, qx, qy, qz) for frames 1 and 2.
    tvec1, tvec2 : array-like of shape (3,)
        Translation vectors for frames 1 and 2.
    trans_threshold : float
        Threshold for minimum significant movement.

    Returns
    -------
    str
        Movement label: "left", "right", "forward", "backward", or "centered".
    """
    # 1. Rotation matrices
    rot1 = R.from_quat([qvec1[1], qvec1[2], qvec1[3], qvec1[0]]).as_matrix()
    rot2 = R.from_quat([qvec2[1], qvec2[2], qvec2[3], qvec2[0]]).as_matrix()

    # 2. Camera centers
    tvec1 = np.array(tvec1)
    tvec2 = np.array(tvec2)
    C1 = -rot1.T @ tvec1
    C2 = -rot2.T @ tvec2

    # 3. World-space translation difference
    trans_diff_world = C2 - C1

    # 4. Project into local camera frame
    trans_diff_local = rot1 @ trans_diff_world

    # 5. Determine movement direction
    x, _, z = trans_diff_local
    if abs(x) > abs(z):
        if abs(x) > trans_threshold:
            return "right" if x > 0 else "left"
    else:
        if abs(z) > trans_threshold:
            return "forward" if z > 0 else "backward"
    return "centered"

def relative_direction_to_point(
    qvec, tvec, target_coord,
    trans_threshold=1.5, secondary_threshold=1.0
):
    """
    Calculate the relative movement direction from a pose to a target 3D coordinate.

    Parameters
    ----------
    qvec : array-like of shape (4,)
        Quaternion (qw, qx, qy, qz) for the current frame.
    tvec : array-like of shape (3,)
        Translation vector for the current frame.
    target_coord : array-like of shape (3,)
        Target 3D coordinate in world frame.
    trans_threshold : float
        Threshold for main axis (x or z) to be considered significant.
    secondary_threshold : float
        Threshold for secondary axis to include combined direction.

    Returns
    -------
    str
        Movement label: e.g. "left", "right", "forward", "backward", 
        or combined directions like "left front", "right back", or "centered".
    """
    # 1. Rotation matrix of current pose
    rot = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

    # 2. Camera center in world coordinates
    tvec = np.array(tvec)
    C = -rot.T @ tvec

    # 3. Vector from camera to target in world space
    trans_diff_world = np.array(target_coord) - C

    # 4. Transform into camera (local) frame
    trans_diff_local = rot @ trans_diff_world
    x, _, z = trans_diff_local

    abs_x, abs_z = abs(x), abs(z)

    # 5. Determine combined direction
    if abs_x > trans_threshold or abs_z > trans_threshold:
        x_dir = "right" if x > 0 else "left"
        z_dir = "forward" if z > 0 else "backward"

        if abs_x > trans_threshold and abs_z > secondary_threshold:
            return f"{x_dir} {z_dir}"
        elif abs_z > trans_threshold and abs_x > secondary_threshold:
            return f"{x_dir} {z_dir}"
        elif abs_x > abs_z:
            return x_dir
        else:
            return z_dir

    return "centered"

def relative_rotation(qvec1, tvec1, qvec2, tvec2, rot_threshold_deg=5.0):
    """
    Calculate the relative rotation between two poses.
    Detects if the person turns around to face the opposite direction.
    """
    # 1. Relative rotation
    r1 = R.from_quat([qvec1[1], qvec1[2], qvec1[3], qvec1[0]])
    r2 = R.from_quat([qvec2[1], qvec2[2], qvec2[3], qvec2[0]])
    r_rel = r2 * r1.inv()

    # 2. Angle magnitude
    rotation_deg = np.degrees(r_rel.magnitude())

    # 3. Determine rotation type
    if rotation_deg < rot_threshold_deg:
        rotation_type = "small rotation"
    else:
        # Analyze Euler angles
        euler_xyz = r_rel.as_euler('xyz', degrees=True)
        roll, pitch, yaw = euler_xyz

        if abs(pitch) > 20 and pitch > 0:
            rotation_type = "looking_down"
        elif abs(pitch) > 20 and pitch < 0:
            rotation_type = "looking_up"
        elif abs(yaw) > 140: # turn to the opposite direction
            rotation_type = "turned_back"
        elif abs(yaw) > 20:
            if yaw > 0:
                rotation_type = "turning_yaw_left"
            else:
                rotation_type = "turning_yaw_right"

    return rotation_deg, rotation_type

def get_relative_movement(distances):
    try:
        trend = np.polyfit(range(len(distances)), distances, 1)[0]  # Linear trend slope
    except np.linalg.LinAlgError:
        return None  # Return None if SVD fails

    if trend < -0.05:  # Threshold for moving closer
        return "closer"
        # return "moving closer"
    elif trend > 0.05:  # Threshold for moving further away
        return "further"
        # return "moving further away"
    else:
        return "stationary"
        # return "remaining relatively stationary"
    
def extract_index(epic_image_name):
    # frame_0000001214.jpg to 1214
    index = int(epic_image_name.split('_')[1].split('.')[0])
    return int(index)

def index_to_filename(index):
    return f"frame_{index:010d}.jpg"

def find_nearby_key_value(number, data_dict, offset=10):
    """
    Searches for a key in the dictionary that matches the given number.
    If an exact match is not found, searches within a specified offset range:
    first checking the next 'offset' numbers, then the previous 'offset' numbers.
    
    Args:
        number: The target number to search for
        data_dict: Dictionary to search in
        offset: Number of positions to search forward and backward (default: 10)
    
    Returns:
        tuple: (value, exact_match_flag) where:
            - value: The corresponding value if found, otherwise -1
            - exact_match_flag: 1 if exact match found, 0 if nearby match found
    """
    if number in data_dict:
        return data_dict[number], 1
    
    # Check the next 'offset' numbers
    total_offset = offset
    for offset in range(1, total_offset+1):
        if number + offset in data_dict:
            return data_dict[number + offset], 0
    
    # Check the previous 'offset' numbers
    for offset in range(1, total_offset+1):
        if number - offset in data_dict:
            return data_dict[number - offset], 0

    return -1, 0 # not found

def find_nearby_key(number, data_dict, offset=10):
    """
    Searches for a key in the dictionary that matches the given number.
    If an exact match is not found, searches within a specified offset range:
    first checking the next 'offset' numbers, then the previous 'offset' numbers.
    
    Args:
        number: The target number to search for
        data_dict: Dictionary to search in
        offset: Number of positions to search forward and backward (default: 10)
    
    Returns:
        tuple: (key, exact_match_flag) where:
            - key: The matching key if found, otherwise -1
            - exact_match_flag: 1 if exact match found, 0 if nearby match found
    """
    if number in data_dict.keys():
        return number, 1
    
    # Check the next 'offset' numbers
    total_offset = offset
    for offset in range(1, total_offset+1):
        current_number = number + offset
        if current_number in data_dict.keys():
            return current_number, 0
    # Check the previous 'offset' numbers
    for offset in range(1, total_offset+1):
        current_number = number - offset
        if current_number in data_dict.keys():
            return current_number, 0

    return -1, 0 # not found


def get_3d_coord_from_2d_segmentation(seg_mask, image_id, colmap_images, colmap_points3D):
    """
    Extracts 3D coordinates from a 2D segmentation mask by finding COLMAP keypoints that fall within the mask.
    
    This function takes a binary segmentation mask and finds all COLMAP 2D keypoints that lie within the 
    segmented region. For each valid keypoint, it retrieves the corresponding 3D point coordinates 
    and returns the average 3D position of all valid points.
    
    Args:
        seg_mask (np.ndarray): Binary mask of shape (H, W) where 1 indicates the segmented object region.
        image_id (int): ID of the image in the COLMAP dataset.
        colmap_images (object): COLMAP image structure containing keypoint information with attributes:
                                - 'xys': 2D keypoint coordinates of shape (N_keypoints, 2)
                                - 'point3D_ids': corresponding 3D point IDs of shape (N_keypoints,)
        colmap_points3D (dict): COLMAP 3D points dictionary mapping point IDs to point structures 
                                with 'xyz' attribute containing 3D coordinates.

    Returns:
        np.ndarray or None: Average 3D coordinate of shape (3,) for the segmented region, 
                           or None if no valid 3D points are found within the mask.
    """
    colmap_image = colmap_images[image_id] 
    # Initialize a list to collect valid 3D points
    collected_xyz = []

    # COLMAP 2D keypoints
    xys = colmap_image.xys  # shape (N_keypoints, 2)
    point3D_ids = colmap_image.point3D_ids  # shape (N_keypoints,)

    # Go through each keypoint
    for i in range(xys.shape[0]):
        x, y = xys[i]
        point3D_id = point3D_ids[i]

        if point3D_id == -1:
            continue  # No corresponding 3D point, skip

        # Round coordinates to integer pixel indices
        x_int = int(round(x))
        y_int = int(round(y))

        # Check if the pixel falls inside the segmentation mask
        if 0 <= x_int < seg_mask.shape[1] and 0 <= y_int < seg_mask.shape[0]:
            if seg_mask[y_int, x_int]:  # seg_mask is (H, W), y first
                if point3D_id in colmap_points3D:
                    xyz = colmap_points3D[point3D_id].xyz
                    collected_xyz.append(xyz)

    if len(collected_xyz) == 0:
        
        return None  # No valid points found

    collected_xyz = np.array(collected_xyz)  # (N_valid, 3)
    avg_3d_coord = np.mean(collected_xyz, axis=0)  # (3,)
    return avg_3d_coord
    
def get_3d_coord_for_all_anchor_objects(segmentation_masks, colmap_images, colmap_points3D):
    """
    Computes 3D coordinates for all anchor objects from their segmentation masks across multiple images.
    
    This function processes segmentation masks for anchor objects (furniture/appliances) and computes
    their average 3D positions by:
    1. For each object, collecting 3D coordinates from all available segmentation masks
    2. Computing the average 3D position across all valid detections
    3. Filtering out objects that don't have valid COLMAP correspondences
    
    Args:
        segmentation_masks (dict): Dictionary mapping object IDs to lists of tuples containing
                                  (image_id, segmentation_mask, metadata) for each detection.
        colmap_images (object): COLMAP image structure containing camera poses and keypoint information.
        colmap_points3D (dict): COLMAP 3D points dictionary mapping point IDs to 3D coordinates.

    Returns:
        dict: Dictionary mapping object IDs to their average 3D coordinates (numpy array of shape (3,)).
              Only includes objects with valid 3D coordinate estimates.
    """
    all_3d_coords = {}

    for obj_id, set_list in segmentation_masks.items():
        coords_list = []
        for image_id, seg_mask, _ in set_list:
            if image_id not in colmap_images:
                print(f"[Warning] image_id {image_id} not found in colmap_images. Skipping object {obj_id}.")
                continue  # Skip if image_id not in COLMAP
            coord = get_3d_coord_from_2d_segmentation(seg_mask, image_id, colmap_images, colmap_points3D)
            if coord is not None:
                coords_list.append(coord)
        if len(coords_list) > 0:
            all_3d_coords[obj_id] = np.mean(coords_list, axis=0)
    return all_3d_coords

def generate_masks_for_all_anchor_objects(visor_to_epic_map, segment_json, mode, rgb_main_dir, output_directory=None, output_resolution=(456, 256), frame_index_to_colmap_imageid_dict=None, colmap_images=None):
    """
    Generates segmentation masks for all anchor objects (furniture/appliances) from VISOR annotations.
    
    This function processes VISOR segmentation annotations to create binary masks for anchor objects
    (furniture and appliances that are not easily movable) by:
    1. Iterating through video annotations and filtering for anchor object classes
    2. Generating binary segmentation masks using the VISOR annotation data
    3. Mapping frame indices to COLMAP image IDs for 3D reconstruction correspondence
    4. Collecting masks for each anchor object across multiple frames
    
    Args:
        visor_to_epic_map (dict): Mapping from VISOR scene/image names to EPIC-KITCHENS image names
        segment_json (dict): VISOR segmentation annotation data containing video_annotations
        mode (str): Dataset mode ("train", "val", or "test")
        rgb_main_dir (str): Base directory containing EPIC-KITCHENS RGB frames
        output_directory (str, optional): Directory to save generated masks. Defaults to None.
        output_resolution (tuple, optional): Resolution for output masks (width, height). Defaults to (456, 256).
        frame_index_to_colmap_imageid_dict (dict, optional): Mapping from frame indices to COLMAP image IDs. Defaults to None.
        colmap_images (dict, optional): COLMAP image data. Defaults to None.

    Returns:
        dict: Dictionary mapping anchor object class IDs to lists of tuples containing
              (image_id, segmentation_mask, improvable_flag) for each detection.
    """
    not_easily_movable_objects_id = [12, 24, 46, 63, 70, 113, 124, 245, 90, 50]
    not_easily_movable_objects_masks = dict()
    
    video_anns = segment_json['video_annotations']
    for tmp_data in video_anns:
        image_name = tmp_data['image']['name']
        scene_id = tmp_data['image']['video']
        epic_image_name = visor_to_epic_map[scene_id][image_name]
        mode = "test" if mode == "val" else mode 
        epic_image_path = os.path.join(rgb_main_dir, mode, scene_id.split('_')[0], scene_id, epic_image_name)
        for tmp_data_ann in tmp_data['annotations']:
            if tmp_data_ann['class_id'] not in not_easily_movable_objects_id:
                continue

            object_name = tmp_data_ann['name'].replace(' ', '_').replace('/', '_')
            class_id = tmp_data_ann['class_id']
            object_epic_image_name = object_name + '_' + epic_image_name
            
            np_mask = generate_masks_for_image(image_name=object_epic_image_name,
                                            image_path=epic_image_path,
                                            masks_info=[tmp_data_ann],
                                            output_directory=output_directory,
                                            output_resolution=output_resolution)
        

            image_index, improvable = find_nearby_key(extract_index(epic_image_name), frame_index_to_colmap_imageid_dict)
            if improvable != 1:
                continue
            image_id = frame_index_to_colmap_imageid_dict[image_index]

            if class_id not in not_easily_movable_objects_masks.keys():
                not_easily_movable_objects_masks[class_id] = [(image_id, np_mask, improvable)]
            else:
                not_easily_movable_objects_masks[class_id].append((image_id, np_mask, improvable))

    return not_easily_movable_objects_masks

def fill_relative_direction_template(
    direction_at_a1=None, direction_at_ak=None,
    template_type="single", object_1="object_1", object_2="object_2",
    a1="a_1", ak="a_k", 
    direction_object1=None, direction_object2=None
):
    """
    Generates question-answer pairs for relative direction queries between two actions.

    This function creates natural language questions and answers about the spatial relationship
    between objects and a person during different actions. It supports both single-object
    scenarios (comparing object position across two actions) and multi-object scenarios
    (comparing positions of two objects during the same action).

    Args:
        direction_at_a1 (str, optional): Direction of object_1 relative to person during action a1.
            Used only for single-object templates. Examples: "left", "right", "front", "back".
        direction_at_ak (str, optional): Direction of object_1 relative to person during action ak.
            Used only for single-object templates. Examples: "left", "right", "front", "back".
        template_type (str): Template type - "single" for one object across two actions,
            "multi" for two objects during one action. Defaults to "single".
        object_1 (str): Name of the first object. Defaults to "object_1".
        object_2 (str): Name of the second object. Used only for multi-object templates.
            Defaults to "object_2".
        a1 (str): Description of the first action. Defaults to "a_1".
        ak (str): Description of the second action. Defaults to "a_k".
        direction_object1 (str, optional): Direction of object_1 during action a1.
            Required for multi-object templates. Examples: "left", "right", "front", "back".
        direction_object2 (str, optional): Direction of object_2 during action a1.
            Required for multi-object templates. Examples: "left", "right", "front", "back".

    Returns:
        tuple: A pair containing (question, answer) where both are natural language strings.
            The question asks about spatial relationships, and the answer provides a detailed
            explanation of the relative positions and any changes that occur.

    Raises:
        ValueError: If template_type is "multi" but direction_object1 or direction_object2
            is not provided.

    Examples:
        >>> fill_relative_direction_template("left", "right", "single", "cup", a1="picking up", ak="putting down")
        ("Does the hand closer to the cup differ when performing `picking up` and `putting down`?",
         "Yes, the hand physically closer to the cup differs between `picking up` and `putting down`...")
    """

    # Support composite directions
    direction_to_text = {
        "left": "to the left of",
        "right": "to the right of",
        "forward": "in front of",
        "backward": "behind",
        "front": "in front of",
        "back": "behind",
        "left front": "to the front-left of",
        "left forward": "to the front-left of",
        "right front": "to the front-right of",
        "right forward": "to the front-right of",
        "left back": "to the back-left of",
        "left backward": "to the back-left of",
        "right back": "to the back-right of",
        "right backward": "to the back-right of"
    }

    def format_direction(direction):
        if direction in direction_to_text:
            return f"{direction_to_text[direction]} the person"
        return f"{direction} of the person"

    def hand_phrase(pos):
        if pos in ['left', 'left front', 'left back', 'left backward', 'left forward']:
            return 'the left hand is physically closer to the object'
        elif pos in ['right', 'right front', 'right back', 'right backward', 'right forward']:
            return 'the right hand is physically closer to the object'
        elif pos in ['front', 'back', 'forward', 'backward']:
            return 'it is unclear which hand is physically closer to the object'
        else:
            return 'it is unclear which hand is physically closer to the object'

    def orientation_note(p1, p2):
        # Normalize for matching
        front_set = {'front', 'left front', 'right front', 'forward'}
        back_set = {'back', 'left back', 'right back', 'backward'}

        if p1 in front_set and p2 in back_set:
            return ' The person turns from facing the object to having their back to it.'
        elif p1 in back_set and p2 in front_set:
            return ' The person turns from having their back to the object to facing it.'
        return ''

    def generate_hand_closer_answer_template(a1, ak, pos1, pos2, obj1):
        h1_phrase = hand_phrase(pos1)
        h2_phrase = hand_phrase(pos2)
        orient_comment = orientation_note(pos1, pos2)

        if h1_phrase != h2_phrase:
            return (f"Yes, the hand physically closer to the {obj1} differs between `{a1}` and `{ak}`. "
                    f"In `{a1}`, the {obj1} is {format_direction(pos1)}, so {h1_phrase}.{orient_comment} "
                    f"In `{ak}`, it is {format_direction(pos2)}, so {h2_phrase}.")
        else:
            return (f"No, the hand physically closer to the {obj1} does not change between `{a1}` and `{ak}`. "
                    f"In both cases, the {obj1} is {format_direction(pos2)}, and {h2_phrase}.{orient_comment}")

    # Map user input direction aliases to standardized ones
    norm_map = {
        "forward": "front", "backward": "back"
    }
    direction_at_a1 = norm_map.get(direction_at_a1, direction_at_a1)
    direction_at_ak = norm_map.get(direction_at_ak, direction_at_ak)
    if direction_object1:
        direction_object1 = norm_map.get(direction_object1, direction_object1)
    if direction_object2:
        direction_object2 = norm_map.get(direction_object2, direction_object2)

    single_templates = [
        f"Does the hand closer to the {object_1} differ when performing `{a1}` and `{ak}`?",
        f"Is the {object_1} {format_direction(direction_at_a1)} when the person is performing `{a1}` and {format_direction(direction_at_ak)} when performing `{ak}`?",
    ]

    multi_templates = [
        f"Are the {object_1} and {object_2} on the same side of the person when performing `{a1}`?",
        f"Is the person facing both the {object_1} and {object_2} from the same side when performing `{a1}`?",
    ]

    if template_type == "single":
        question = random.choice(single_templates)
        if direction_at_a1 == direction_at_ak:
            same_direction_phrases = [
                f"The {object_1} remains {format_direction(direction_at_a1)} during both `{a1}` and `{ak}`.",
                f"The {object_1} is seen {format_direction(direction_at_a1)} at the beginning. During {ak}, it remains {format_direction(direction_at_ak)}.",
                f"The {object_1} remains consistently {format_direction(direction_at_a1)} during both '{a1}' and '{ak}'.",
                f"The {object_1} does not change its position relative to the person; it is {format_direction(direction_at_a1)} in both '{a1}' and '{ak}'."
            ]
            answer = random.choice(same_direction_phrases)
        else:
            transition_phrases = [
                f"Initially, the {object_1} is {format_direction(direction_at_a1)}, but as the person moves, it becomes {format_direction(direction_at_ak)}.",
                f"At first, the {object_1} appears {format_direction(direction_at_a1)}, but during `{ak}`, it appears {format_direction(direction_at_ak)} due to the person's movement.",
                f"The {object_1} is seen {format_direction(direction_at_a1)} at the beginning. During {ak}, it shifts to {format_direction(direction_at_ak)} relative to the person."
                f"From the person's initial viewpoint, the {object_1} is {format_direction(direction_at_a1)}. As they move, their perspective changes, and the object becomes {format_direction(direction_at_ak)}."
            ]
            answer = random.choice(transition_phrases)
        if "Does the hand closer to the" in question:
            answer = generate_hand_closer_answer_template(a1, ak, direction_at_a1, direction_at_ak, object_1)

    elif template_type == "multi":
        if direction_object1 is None or direction_object2 is None:
            raise ValueError("For 'multi' template, direction_object1 and direction_object2 must be provided.")

        question = random.choice(multi_templates)
        if direction_object1 == direction_object2:
            answer = f"Yes, both the {object_1} and {object_2} are {format_direction(direction_object1)} when the person is performing `{a1}`."
        else:
            answer = f"No, the {object_1} is {format_direction(direction_object1)}, while the {object_2} is {format_direction(direction_object2)} when the person is performing `{a1}`."

    return question, answer


def fill_relative_distance_template(
    relative_movement=None,
    template_type="single",
    object_1="object_1",
    object_2="object_2",
    a1="a_1",
    ak="a_k",
    closer_object_label="object_1",
):
    """
    Fills a relative distance QA template and generates a natural answer.

    Args:
        relative_movement (str): Movement relative to object_1 between <a_1> and <a_k> ("closer", "further", "stationary") (only for single-object)
        template_type (str): "single" or "multi"
        object_1 (str): First object name
        object_2 (str): Second object name (only for multi-object)
        a1 (str): Action narration for action 1
        ak (str): Action narration for action k
        closer_object_label (str): Which object is closer to the person during a1 ("object_1", "object_2", or "similar") (only for multi-object)

    Returns:
        (str, str): Filled-in question, natural answer
    """

    single_templates = [
        f"Does the person move closer to the {object_1} between `{a1}` and `{ak}`?",
        f"Does the person move away from the {object_1} between `{a1}` and `{ak}`?",
        f"Does the person end up closer to the {object_1} after performing `{ak}`?",
        f"Is the person closer to the {object_1} when `{a1}` or when `{ak}`?",
        f"During which action is the person closest to the {object_1}?"
    ]

    multi_templates = [
        f"During `{a1}`, is the person closer to the {object_1} than to the {object_2}?",
        f"During `{a1}`, would it be easier for the person to access the {object_1} than to the {object_2}?"
    ]

    if template_type == "single":
        question = random.choice(single_templates)
    elif template_type == "multi":
        question = random.choice(multi_templates)
    else:
        raise ValueError("template_type must be either 'single' or 'multi'")

    # Generate answer
    if template_type == "single":
        if relative_movement == "stationary":
            answer = f"The person remains at about the same distance from the {object_1} when performing both `{a1}` and `{ak}`."
        elif relative_movement == "closer":
            transition_phrases = [
                f"The person moves closer to the {object_1} from `{a1}` to `{ak}`.",
                f"The person starts off farther from the {object_1} when performing `{a1}`, but ends up closer to it after performing `{ak}`.",
                f"The person approaches the {object_1} while moving from `{a1}` to `{ak}`."
            ]
            answer = random.choice(transition_phrases)
        elif relative_movement == "further":
            transition_phrases = [
                f"The person moves further away from the {object_1} from `{a1}` to `{ak}`.",
                f"The person starts off closer to the {object_1} at `{a1}`, but ends up farther from it after `{ak}`.",
                f"The person moves away from the {object_1} while moving from `{a1}` to `{ak}`."
            ]
            answer = random.choice(transition_phrases)
        else:
            answer = "Movement information is not available."

    elif template_type == "multi":
        if closer_object_label == object_1:
            answer = f"Yes, the person is closer to the {object_1} than to the {object_2} when performing `{a1}`."
        elif closer_object_label == object_2:
            answer = f"No, the person is closer to the {object_2} than to the {object_1} when performing `{a1}`."
        elif closer_object_label == "similar":
            answer = f"The person is at a similar distance from both the {object_1} and the {object_2} when performing `{a1}`."
        else:
            answer = f"The person's relative distance to {object_1} and {object_2} is unclear when performing `{a1}`."

    return question, answer


def fill_find_my_item_qwen_template(object_1, action_name, object_direction, question_type="location"):

    """
    Generates a prompt for Qwen-VL model to generate natural language answers to 'find my item' questions.
    
    This function creates structured prompts that guide the Qwen-VL model to answer questions about
    object locations based on video content and known directional information. The prompts are designed
    to handle two types of questions:
    1. "location" - asking where an object is currently located and how to reach it
    2. "after_action" - asking where an object was left after a specific action and how to retrieve it
    
    The function emphasizes that the video shows past actions and the model should use the provided
    directional information rather than inferring directions from the video content.

    Args:
        object_1 (str): Name of the object that was placed (e.g., "plate", "cup", "book")
        action_name (str): Description of the action performed (e.g., "put down the plate", "place the cup")
        object_direction (str): Current direction of the object relative to the person.
            Must be one of: 'left', 'right', 'forward', 'backward'
        question_type (str): Type of question to generate prompt for.
            Must be either "location" or "after_action". Defaults to "location".

    Returns:
        str: A formatted prompt string designed for the Qwen-VL model that includes:
            - Context about the video content and action
            - Clear instructions about using provided directional information
            - Specific guidelines for generating natural language responses
            - Constraints to avoid video references and invented details

    Raises:
        ValueError: If question_type is not "location" or "after_action"

    Examples:
        >>> fill_find_my_item_qwen_template("plate", "put down the plate", "left", "location")
        # Returns a prompt asking where the plate is and how to reach it
        
        >>> fill_find_my_item_qwen_template("cup", "place the cup", "forward", "after_action") 
        # Returns a prompt asking where the cup was left after placing it
    """

    # Format direction for natural language
    if object_direction == "forward":
        direction_phrase = "in front of the person"
    elif object_direction == "backward":
        direction_phrase = "behind the person"
    else:
        direction_phrase = f"to the {object_direction} of the person"

    if question_type == "location":
        question = f"Where is the {object_1}, and how can the person get to it?"
        prompt = f"""You are given a short video showing the action: {action_name}.
    In the video, the person places the object: {object_1}.

    Note: The video only shows the **past action** — the moment the object was last placed. 
    You, the assistant, are currently **not in the video**. The person's current position is unknown from the video.
    So, do not try to estimate where the person is now from the video.

    However, you are told the object is now located **{direction_phrase}** from the person's current position. 
    This direction is accurate and must be used to answer the question.

    Now answer the following question:

    "{question}"

    Your answer must:
    - First describe the surroundings around the object at the last moment it was visible (based only on the video)
    - Then, use the known direction "{direction_phrase}" to describe where the object is now and how the person can reach it
    - Do not guess or infer directions from the video
    - Do not mention the video directly
    - Do not invent room layouts or paths
    - Only write one fluent, natural English sentence
    """

    elif question_type == "after_action":
        question = f"After performing {action_name}, where did the person leave the {object_1} and how to reach it?"
        prompt = f"""You are given a short video showing the action: {action_name}.
    In this video, the person places the object: {object_1}.

    ⚠️ Important:
    - The video only shows the **past action** of placing the object.
    - You are **not currently in the video**.
    - The person's current position is **after** the video ends, and **not visible**.
    - You are told the object is now located at: **{direction_phrase}**
    This direction is accurate and must be used exactly as given.

    First, briefly describe the surroundings where the object was placed at the **end of the action** (based on the last moment in the video).

    Then, using the known direction "{direction_phrase}", describe where the object is now and how the person can reach it.

    Do NOT:
    - Guess or infer directions from the video
    - Use phrases like “to the right” or “in front” unless they match "{direction_phrase}"
    - Mention the video directly
    - Invent room layouts or walking paths

    Write only **one fluent and natural English sentence** that combines both parts.
    """
    else:
        raise ValueError("question_type must be 'location' or 'after_action'")

    return prompt



def fill_find_my_item_template(object_1, 
                               direction, 
                               predictor, 
                               action_name, 
                               object_1_3d_coord, 
                               anchor_objects_3d_coords_dict, 
                               qwen_start_frame, 
                               qwen_stop_frame,
                               rgb_main_dir,
                               video_id,
                               rgb_mode="train",
                               uniform_sample_max_frames=10,):
    """
    Generates question-answer pairs for object location and navigation queries.

    This function creates natural language questions and answers about object locations,
    post-action object positions, and spatial relationships between objects. It supports
    three types of queries:
    1. Current object location and navigation instructions
    2. Object position after performing a specific action
    3. Distance-based comparisons between objects and anchor points

    The function uses a multimodal predictor (Qwen) to generate contextual answers
    based on video frames and spatial information.

    Args:
        object_1 (str): Name of the target object to locate.
        direction (str): Relative direction of the object to the person 
            ('left', 'right', 'forward', 'backward').
        predictor: Multimodal model for generating contextual answers.
        action_name (str): Name of the action performed before the query.
        object_1_3d_coord (np.ndarray): 3D coordinates of the target object.
        anchor_objects_3d_coords_dict (dict): Dictionary mapping anchor object IDs 
            to their 3D coordinates for spatial comparisons.
        qwen_start_frame (int): Starting frame index for video sampling.
        qwen_stop_frame (int): Ending frame index for video sampling.
        rgb_main_dir (str): Base directory containing RGB video frames.
        video_id (str): Unique identifier for the video sequence.
        rgb_mode (str, optional): Dataset mode for frame loading. Defaults to "train".
        uniform_sample_max_frames (int, optional): Maximum number of frames to sample 
            uniformly from the video. Defaults to 10.

    Returns:
        tuple: A pair containing (question, answer) where both are natural language strings.
            The question asks about object location or spatial relationships, and the answer
            provides contextual information based on video content and spatial data.

    Raises:
        ValueError: If anchor_objects_3d_coords_dict is None but distance comparison
            question is selected.
    """

    if anchor_objects_3d_coords_dict is not None:
        random_ids = random.sample(list(anchor_objects_3d_coords_dict.keys()), 2)
        anchor_object_1, anchor_object_2 = [anchor_objects_id_to_name_dict[i] for i in random_ids]
        anchor_object_1_3d_coord = anchor_objects_3d_coords_dict[random_ids[0]]
        anchor_object_2_3d_coord = anchor_objects_3d_coords_dict[random_ids[1]]

    question_templates = [
        f"Where is the {object_1}, and how can the person get to it?",
        f"After performing `{action_name}`, where did the person leave the {object_1} and how to reach it?",
        f"Would it be closer for the person to bring the {object_1} to the {anchor_object_1} or the {anchor_object_2}?"
    ]

    images = sample_uniform_frames(qwen_start_frame, qwen_stop_frame, num_samples=uniform_sample_max_frames, extract_index=False)
    images = [os.path.join(rgb_main_dir, rgb_mode, video_id.split('_')[0], video_id, img) for img in images]
    
    if anchor_objects_3d_coords_dict is not None:
        question = random.choice(question_templates)
    else:
        question = random.choice(question_templates[:-1])  # Exclude the last question if no anchor objects

    if "Where is the" in question:
        question_type = "location"
        text_query = fill_find_my_item_qwen_template(
            object_1=object_1,
            action_name=action_name,
            object_direction=direction,
            question_type=question_type
        )
        prompt = predictor.create_prompt(text_query, images, type='image', downsample_factor=1)
        inputs = predictor.run_tokenizer(prompt=prompt)
        answer = predictor.run_inference(inputs=inputs)

    elif "After performing" in question:
        question_type = "after_action"
        text_query = fill_find_my_item_qwen_template(
            object_1=object_1,
            action_name=action_name,
            object_direction=direction,
            question_type=question_type
        )
        prompt = predictor.create_prompt(text_query, images, type='image', downsample_factor=1)
        inputs = predictor.run_tokenizer(prompt=prompt)
        answer = predictor.run_inference(inputs=inputs)

    elif "Would it be closer" in question:
        distance_to_anchor_1 = np.linalg.norm(object_1_3d_coord - anchor_object_1_3d_coord)
        distance_to_anchor_2 = np.linalg.norm(object_1_3d_coord - anchor_object_2_3d_coord)
        if abs(distance_to_anchor_1) - abs(distance_to_anchor_2) > 0.5:
            answer = f"It would be closer for the person to bring the {object_1} to the {anchor_object_2}."
        elif abs(distance_to_anchor_1) - abs(distance_to_anchor_2) < -0.5:
            answer = f"It would be closer for the person to bring the {object_1} to the {anchor_object_1}."
        else:
            answer = f"The {anchor_object_1} and the {anchor_object_2} are about the same distance from the person."

    return question, answer



def generate_relative_direction_qa(action_indice, df_Pid, frame_index_to_colmap_imageid_dict, colmap_images, anchor_objects_3d_coords_dict=None):
    # input: a list of action label index, dict[frame_id] = colamp_index, df_Pid, colmap_images, number of anchor objects (1 or 2)
    # output: dict of qa_type, query start frame, query stop frame, question, answer 
    # import pdb; pdb.set_trace() # ! DEBUG
    a_1_index = random.choice(action_indice[:len(action_indice) // 2 - 1])
    a_2_index = random.choice(action_indice[len(action_indice) // 2:])
    
    a_1_row = df_Pid.iloc[a_1_index]
    a_2_row = df_Pid.iloc[a_2_index]
    a_1_start_frame_idx, a_1_stop_frame_idx = a_1_row['start_frame'], a_1_row['stop_frame']
    a_2_start_frame_idx, a_2_stop_frame_idx = a_2_row['start_frame'], a_2_row['stop_frame']
    
    a_1_mid_frame_idx = (a_1_start_frame_idx + a_1_stop_frame_idx) // 2
    a_2_mid_frame_idx = (a_2_start_frame_idx + a_2_stop_frame_idx) // 2

    # import pdb; pdb.set_trace() # ! DEBUG
    # may not have the exact frame with camera pose
    a_1_colmap_image_id, _ = find_nearby_key_value(a_1_mid_frame_idx, frame_index_to_colmap_imageid_dict)
    a_2_colmap_image_id, _ = find_nearby_key_value(a_2_mid_frame_idx, frame_index_to_colmap_imageid_dict)
    if a_1_colmap_image_id == -1 or a_2_colmap_image_id == -1:
        return {}    # ! TODO need to handle this case
    
    a_1_colmap_image = colmap_images[a_1_colmap_image_id]
    a_2_colmap_image = colmap_images[a_2_colmap_image_id]

    a_1_qvec = a_1_colmap_image.qvec
    a_1_tvec = a_1_colmap_image.tvec
    a_2_qvec = a_2_colmap_image.qvec
    a_2_tvec = a_2_colmap_image.tvec
    
    template_type = random.choice(["single", "multi"])
    if len(list(anchor_objects_3d_coords_dict.keys())) < 2 or template_type == "single":
        template_type = "single"
        object_ids = random.sample(list(anchor_objects_3d_coords_dict.keys()), 1)
        object_1_name = anchor_objects_id_to_name_dict[object_ids[0]]
        object_1_3d_coord = anchor_objects_3d_coords_dict[object_ids[0]]
    else: 
        object_ids = random.sample(list(anchor_objects_3d_coords_dict.keys()), 2)
        object_1_name = anchor_objects_id_to_name_dict[object_ids[0]]
        object_2_name = anchor_objects_id_to_name_dict[object_ids[1]]
        object_1_3d_coord = anchor_objects_3d_coords_dict[object_ids[0]] # ! TODO  add the 3D coord of the object_1
        object_2_3d_coord = anchor_objects_3d_coords_dict[object_ids[1]] # ! TODO  add the 3D coord of the object_2

    a1_naration = a_1_row['narration']
    ak_naration = a_2_row['narration']
    if template_type == "single":
        # relative direction
        direction_at_a1 = relative_direction_to_point(a_1_qvec, a_1_tvec, object_1_3d_coord)
        direction_at_ak = relative_direction_to_point(a_2_qvec, a_2_tvec, object_1_3d_coord)
        # import pdb; pdb.set_trace() # ! DEBUG
        question, answer = fill_relative_direction_template(
            direction_at_a1=direction_at_a1,
            direction_at_ak=direction_at_ak,
            template_type=template_type,
            object_1=object_1_name, object_2=None,
            a1=a1_naration, ak=ak_naration
        )

        return {
            "qa_type": "relative_direction",
            "template_type": template_type,
            "query_start_frame": a_1_start_frame_idx,
            "query_stop_frame": a_2_stop_frame_idx,
            "question": question,
            "answer": answer,
            "object_1": object_1_name,
            }
    
    elif template_type == "multi":
        direction_to_object_1_at_a1 = relative_direction_to_point(a_1_qvec, a_1_tvec, object_1_3d_coord)
        direction_to_object_2_at_a1 = relative_direction_to_point(a_2_qvec, a_2_tvec, object_2_3d_coord)
        # import pdb; pdb.set_trace() # ! DEBUG
        question, answer = fill_relative_direction_template(
            direction_object1=direction_to_object_1_at_a1,
            direction_object2=direction_to_object_2_at_a1,
            template_type=template_type,
            object_1=object_1_name, object_2=object_2_name,
            a1=a1_naration
        )

        return {
            "qa_type": "relative_direction",
            "template_type": template_type,
            "query_start_frame": a_1_start_frame_idx,
            "query_stop_frame": a_2_stop_frame_idx,
            "question": question,
            "answer": answer,
            "object_1": object_1_name,
            "object_2": object_2_name
            }

    raise ValueError("template_type must be either 'single' or 'multi'")

def fill_furniture_affordance_prediction_qwen_template(options, previous_actions, groundtruth_anchor_object, options_relative_movement):
    """
    Generates a prompt template for furniture affordance prediction using Qwen model.
    
    This function creates a structured prompt that combines information about previous actions,
    current movement patterns, and available object options to predict which object a person
    will most likely interact with next. The prompt is designed to guide the Qwen model to
    generate natural, human-like responses about affordance prediction.
    
    Args:
        options (list): List of available object options for interaction
        previous_actions (list): List of previous actions performed by the person
        groundtruth_anchor_object (str): The correct object that the person will interact with
        options_relative_movement (dict): Dictionary mapping objects to their relative movement descriptions
        
    Returns:
        str: Formatted prompt string for the Qwen model
    """
    
    previous_actions_text = " -> ".join(previous_actions)

    # Build movement descriptions
    movement_descriptions = []
    for obj in options:
        if obj in options_relative_movement:
            movement = options_relative_movement[obj]
            movement_descriptions.append(f"The person is moving {movement} to the {obj}.")
    movement_text = " ".join(movement_descriptions)

    # Build options text
    options_text = ", ".join(options)

    prompt = f"""You are given information about a person and their surroundings. 
    Previous actions performed by the person: {previous_actions_text}.
    Movement relative to nearby objects: {movement_text}
    The available options are: {options_text}.
    Based on the previous actions and current movements, please generate a natural, fluent English sentence that answers the following question:
    "Which object will the person most likely interact with next?"
    The answer should reflect that the person is most likely to interact with the {groundtruth_anchor_object}. Do not mention the list of options directly in your answer. Focus on explaining naturally why the person is approaching or likely to interact with the {groundtruth_anchor_object} based on their actions and movement. Keep the response concise and natural, as if you are a human describing the situation.
    Do not repeat the question or add unrelated commentary.
    """
    return prompt

def fill_furniture_affordance_prediction_template(narrations,
                                                  options, 
                                                  predictor, 
                                                  groundtruth_anchor_object, 
                                                  options_relative_movement, 
                                                  query_video_start_frame, 
                                                  query_video_stop_frame,
                                                  rgb_main_dir,
                                                  video_id,
                                                  uniform_sample_max_frames=10,
                                                  rgb_mode="train"):
    """
    Generates question-answer pairs for furniture affordance prediction tasks.
    
    This function creates multiple-choice questions about which object a person will most likely
    interact with next, based on their previous actions and current movement patterns. It samples
    video frames uniformly across the query time range and uses a vision-language model to generate
    natural language answers about affordance prediction.
    
    Args:
        narrations (list): List of action narrations describing previous behaviors
        options (list): List of available object options for the multiple choice question
        predictor: Vision-language model predictor for generating answers
        groundtruth_anchor_object (str): The correct object that the person will interact with
        options_relative_movement (dict): Dictionary mapping objects to movement descriptions
        query_video_start_frame (int): Starting frame index for video sampling
        query_video_stop_frame (int): Ending frame index for video sampling
        rgb_main_dir (str): Base directory containing RGB video frames
        video_id (str): Identifier for the current video
        uniform_sample_max_frames (int, optional): Maximum number of frames to sample. Defaults to 10.
        rgb_mode (str, optional): Dataset mode ("train", "val", "test"). Defaults to "train".
        
    Returns:
        tuple: A pair containing (question, answer) where question is a multiple-choice format
               and answer is the model-generated response about affordance prediction
    """

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # Question templates
    question_templates = [
        "Considering the person's previous actions and the current movement, which object will they most likely interact with next?",
        "Which of the following objects does the person interact with next, given their previous actions and current motion?",
        "Based on what the person has done so far and how they're moving now, which nearby object is the person preparing to interact with?"
    ]

    options_str = ", ".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options)])

    # Randomly select a question template
    question = random.choice(question_templates)

    question = question + "\n" + options_str

    text_query = fill_furniture_affordance_prediction_qwen_template(
        options=options,
        previous_actions=narrations,
        groundtruth_anchor_object=groundtruth_anchor_object,
        options_relative_movement=options_relative_movement
    )

    images = sample_uniform_frames(query_video_start_frame, query_video_stop_frame, num_samples=uniform_sample_max_frames, extract_index=False)
    images = [os.path.join(rgb_main_dir, rgb_mode, video_id.split('_')[0], video_id, img) for img in images]

    # Use the predictor to get the answer
    prompt = predictor.create_prompt(text_query, images, type='image', downsample_factor=1)
    inputs = predictor.run_tokenizer(prompt=prompt)

    # Set answer to None
    answer = predictor.run_inference(inputs=inputs)  # ! TODO add the answer using qwen

    print(f'question: {question}, \n answer: {answer} \n groundtruth_anchor_object: {groundtruth_anchor_object}')
    

    return question, answer

def sample_frames_with_camera_pose(start, end, num_frames, frame_index_to_colmap_imageid_dict):
    """
    Sample frames uniformly between start and end that exist in frame_index_to_colmap_imageid_dict (image index registered to colmap).

    Args:
        start (int): Start frame number (inclusive).
        end (int): End frame number (inclusive).
        num_frames (int): Target number of frames to sample.
        frame_index_to_colmap_imageid_dict (dict): Dictionary of valid frame indices.

    Returns:
        list[int]: List of sampled frame indices.
    """

    # 1. Find all valid frames between start and end
    valid_frames = [i for i in range(start, end + 1) if i in frame_index_to_colmap_imageid_dict]

    # 2. If less than 50 valid frames, just return all
    if len(valid_frames) <= 50:
        return valid_frames

    # 3. Otherwise, sample uniformly
    if len(valid_frames) <= num_frames:
        return valid_frames

    # Compute equally spaced indices
    sampled_indices = np.linspace(0, len(valid_frames) - 1, num=num_frames, dtype=int)
    sampled_frames = [valid_frames[i] for i in sampled_indices]

    return sampled_frames

def sample_uniform_frames(start_filename, end_filename, num_samples=10, extract_index=True):

    def extract_frame_number(filename):
        return int(filename.split('_')[1].split('.')[0])
    def construct_frame_filename(frame_number):
        return f"frame_{frame_number:010d}.jpg"
    
    if extract_index:
        start_frame = extract_frame_number(start_filename)
        end_frame = extract_frame_number(end_filename)
    else:
        start_frame = start_filename
        end_frame = end_filename

    # Generate 10 uniformly spaced frame indices (including start and end)
    sampled_frames = np.linspace(start_frame, end_frame, num=num_samples, dtype=int)

    # Convert to filename format
    return [construct_frame_filename(f) for f in sampled_frames]

def generate_relative_distance_qa(action_indice, df_Pid, frame_index_to_colmap_imageid_dict, colmap_images, anchor_objects_3d_coords_dict=None):
    # input: a list of action label index, dict[frame_id] = colamp_index, df_Pid, colmap_images, number of anchor objects (1 or 2)
    a_1_index = random.choice(action_indice[:len(action_indice) // 2 - 1])
    a_2_index = random.choice(action_indice[len(action_indice) // 2:])
    
    a_1_row = df_Pid.iloc[a_1_index]
    a_2_row = df_Pid.iloc[a_2_index]
    a_1_start_frame_idx, a_1_stop_frame_idx = a_1_row['start_frame'], a_1_row['stop_frame']
    a_2_start_frame_idx, a_2_stop_frame_idx = a_2_row['start_frame'], a_2_row['stop_frame']
    
    a_1_mid_frame_idx = (a_1_start_frame_idx + a_1_stop_frame_idx) // 2
    a_2_mid_frame_idx = (a_2_start_frame_idx + a_2_stop_frame_idx) // 2

    num_frames_to_sample = len(action_indice) * 0   # ! DEBUG
    sampled_frames = sample_frames_with_camera_pose(a_1_mid_frame_idx, a_2_mid_frame_idx, num_frames_to_sample, frame_index_to_colmap_imageid_dict)
    # print(f'num of frames to sample: {len(sampled_frames)}')
    template_type = random.choice(["single", "multi"])
    
# anchor objects
    if len(list(anchor_objects_3d_coords_dict.keys())) < 2 or template_type == "single":
        template_type = "single"
        object_ids = random.sample(list(anchor_objects_3d_coords_dict.keys()), 1)
        object_1_name = anchor_objects_id_to_name_dict[object_ids[0]]
        object_1_3d_coord = anchor_objects_3d_coords_dict[object_ids[0]]
    else:
        object_ids = random.sample(list(anchor_objects_3d_coords_dict.keys()), 2)
        object_1_name = anchor_objects_id_to_name_dict[object_ids[0]]
        object_2_name = anchor_objects_id_to_name_dict[object_ids[1]]
        object_1_3d_coord = anchor_objects_3d_coords_dict[object_ids[0]] # ! TODO  add the 3D coord of the object_1
        object_2_3d_coord = anchor_objects_3d_coords_dict[object_ids[1]] # ! TODO  add the 3D coord of the object_2

    a1_naration = a_1_row['narration']
    ak_naration = a_2_row['narration']
    if template_type == "single":
        distances = []
        for frame in sampled_frames:
            image_id = frame_index_to_colmap_imageid_dict[frame]
            colmap_image = colmap_images[image_id]
            frame_tvec = colmap_image.tvec
            object_coord = object_1_3d_coord # ! TODO  add the 3D coord of the object_1
            displacement_world = object_coord - frame_tvec
            distance = np.linalg.norm(displacement_world).item()
            distances.append(distance)
        if len(distances) == 0:
            return {}
        relative_movement = get_relative_movement(distances)
        if relative_movement == None:
            return {}
        a1_naration = a_1_row['narration']
        question, answer = fill_relative_distance_template(
            relative_movement=relative_movement,
            template_type=template_type,
            object_1=object_1_name, object_2=None,
            a1=a1_naration, ak=ak_naration,
            closer_object_label=None
        )
        return {   
            "qa_type": "relative_distance",
            "template_type": template_type,
            "query_start_frame": a_1_start_frame_idx,
            "query_stop_frame": a_2_stop_frame_idx,
            "question": question,
            "answer": answer,
            "object_1": object_1_name
        }
    
    elif template_type == "multi":
        a_1_mid_image_id, _ = find_nearby_key_value(a_1_mid_frame_idx, frame_index_to_colmap_imageid_dict, offset=1)  # ! DEBUG
        if a_1_mid_image_id == -1 or a_1_mid_image_id not in colmap_images.keys():
            return {}
        object_1_distance = object_1_3d_coord - colmap_images[a_1_mid_image_id].tvec
        object_2_distance = object_2_3d_coord - colmap_images[a_1_mid_image_id].tvec
        object_1_distance = np.linalg.norm(object_1_distance).item()
        object_2_distance = np.linalg.norm(object_2_distance).item()
        if abs(object_1_distance) - abs(object_2_distance) > 0.1:  # ! TODO need to check distance threshold
            closer_object_label = object_2_name
        elif abs(object_1_distance) - abs(object_2_distance) < -0.1:
            closer_object_label = object_1_name
        else: 
            closer_object_label = "similar"

        question, answer = fill_relative_distance_template(
            relative_movement=None,
            template_type=template_type,
            object_1=object_1_name, object_2=object_2_name,
            a1=a1_naration, ak=None,
            closer_object_label=closer_object_label
        )
        return {   
            "qa_type": "relative_distance",
            "template_type": template_type,
            "query_start_frame": a_1_start_frame_idx,
            "query_stop_frame": a_2_stop_frame_idx,
            "question": question,
            "answer": answer,
            "object_1": object_1_name,
            "object_2": object_2_name
        }
    

def generate_object_motion_prompt(answer: str) -> str:
    """
    Generate a prompt for a vision-language model to check if an object in the video has been moved,
    and if so, to describe the movement and integrate it into the original answer.

    Parameters:
    - answer (str): The original two-sentence answer. The first sentence describes the object’s location.
                    The second sentence describes how the person moves to obtain the object.
    - video_path (str): The path or identifier of the video that may contain the object movement.

    Returns:
    - prompt (str): A full prompt for the vision-language model.
    """
    prompt = f"""You are given a short video clip.

This video may contain a person interacting with or moving an object. Your task is to analyze whether the object mentioned in the following answer has been moved from its original position.

Here is the current answer:
"{answer}"

Step 1: Determine whether the object mentioned was moved after the original placement.
Step 2: If the object was moved or altered, describe clearly how it was changed. This includes physical movement (e.g., direction, distance, and new location if possible), structural changes (e.g., breaking or disassembly), or separation from other items it was originally grouped with (e.g., a bowl and its contents being separated).
Step 3: Append a new sentence to the answer that describes this movement, so the updated answer reflects the current status.

Return ONLY the updated answer with the movement description appended (if any). If there is no movement, return the original answer unchanged."""

    return prompt

def generate_find_my_item_qa(action_indice, 
                             df_Pid, 
                             predictor, 
                             frame_index_to_colmap_imageid_dict, 
                             colmap_images, 
                             rgb_main_dir,
                             video_id,
                             rgb_mode="train",
                             uniform_sample_max_frames=10,
                             anchor_objects_3d_coords_dict=None,
                             object_3d_coord=None):
    
    
    item_occurred_action_index = action_indice[0]
    find_action_index = action_indice[-1]
    find_action_row = df_Pid.iloc[find_action_index]
    object_name = df_Pid.iloc[item_occurred_action_index]['noun']

    # ! DEBUG
    object_class_id = df_Pid.iloc[item_occurred_action_index]['noun_class']
    if object_class_id not in small_objects_id_to_name_dict.keys():
        return {}
    # ! DEBUG END
    
    a_1_start_frame_idx, a_1_stop_frame_idx = find_action_row['start_frame'], find_action_row['stop_frame']
    a_1_mid_frame_idx = (a_1_start_frame_idx + a_1_stop_frame_idx) // 2
    a_1_mid_frame_idx, _ = find_nearby_key_value(a_1_mid_frame_idx, frame_index_to_colmap_imageid_dict, offset=1)  # ! DEBUG
    
    if a_1_mid_frame_idx == -1 or a_1_mid_frame_idx not in frame_index_to_colmap_imageid_dict.keys():
        print('quit in 1008 utils.py')
        return {}
    
    a_1_colmap_image_id = frame_index_to_colmap_imageid_dict[a_1_mid_frame_idx]
    a_1_colmap_image = colmap_images[a_1_colmap_image_id]
    a_1_qvec = a_1_colmap_image.qvec
    a_1_tvec = a_1_colmap_image.tvec
    direction = relative_direction_to_point(a_1_qvec, a_1_tvec, object_3d_coord, 2.0, 1.8)
    
    qwen_start_frame = df_Pid.iloc[item_occurred_action_index]['start_frame'].item()
    qwen_stop_frame = df_Pid.iloc[item_occurred_action_index]['stop_frame'].item()
    question, answer = fill_find_my_item_template(
        object_1=object_name,
        direction=direction,
        predictor=predictor,
        action_name=find_action_row['narration'],
        object_1_3d_coord=object_3d_coord,
        anchor_objects_3d_coords_dict=anchor_objects_3d_coords_dict,
        qwen_start_frame=qwen_start_frame,
        qwen_stop_frame=qwen_stop_frame,
        rgb_main_dir=rgb_main_dir,
        video_id=video_id,
        rgb_mode=rgb_mode,
        uniform_sample_max_frames=uniform_sample_max_frames
    )
    print(f'question: {question}, \nanswer: {answer} \nobject_name: {object_name}')

    # ! DEBUG to prevent intermediate movements of the query object
    if "Would it be closer" not in question:
        query_start_frame = df_Pid.iloc[action_indice[0]]['start_frame']
        query_stop_frame = df_Pid.iloc[action_indice[-1]]['stop_frame']
        sample_frames = sample_frames_with_camera_pose(query_start_frame, query_stop_frame, 10, frame_index_to_colmap_imageid_dict)

        sample_frames_paths = []
        for frame in sample_frames:
            image_id = frame_index_to_colmap_imageid_dict[frame]
            image_name = colmap_images[image_id].name
            image_path = os.path.join(rgb_main_dir, rgb_mode, video_id.split('_')[0], video_id, image_name)
            sample_frames_paths.append(image_path)

        object_transition_prompt = generate_object_motion_prompt(answer)
        prompt = predictor.create_prompt(object_transition_prompt, sample_frames_paths, type='image', downsample_factor=1)
        inputs = predictor.run_tokenizer(prompt=prompt)
        answer = predictor.run_inference(inputs=inputs)
        

    return {
        "qa_type": "find_my_item",
        "template_type": "single",
        "query_start_frame": df_Pid.iloc[action_indice[0]]['start_frame'],
        "query_stop_frame": df_Pid.iloc[action_indice[-1]]['stop_frame'],
        "question": question,
        "answer": answer,
        "object_1": object_name
    }



def generate_furniture_affordance_qa(action_indice, 
                                    df_Pid, 
                                    frame_index_to_colmap_imageid_dict, 
                                    colmap_images, 
                                    groundtruth_anchor_object, 
                                    list_of_options_3d_coord, 
                                    predictor, 
                                    video_id,
                                    rgb_main_dir,
                                    list_of_options=['fridge', 'microwave', 'sink', 'hob'],
                                    uniform_sample_max_frames=10,
                                    rgb_mode="train"):

    if groundtruth_anchor_object not in list_of_options:
        raise ValueError(f"groundtruth_anchor_object must be one of {list_of_options}")
    print(f'action_indice: {action_indice}')

    narrations = [df_Pid.iloc[i]['narration'] for i in action_indice]
    query_video_start_frame = df_Pid.iloc[action_indice[0]]['start_frame']
    query_video_stop_frame = df_Pid.iloc[action_indice[-1]]['stop_frame']
    num_samples = len(action_indice) * 5
    sampled_frames = sample_frames_with_camera_pose(query_video_start_frame, query_video_stop_frame, num_samples, frame_index_to_colmap_imageid_dict)
    
    options_distances = {key: [] for key in list_of_options}
    options_relative_movement = {key: "" for key in list_of_options}

    for frame in sampled_frames:
        image_id = frame_index_to_colmap_imageid_dict[frame]
        colmap_image = colmap_images[image_id]
        frame_tvec = colmap_image.tvec
        for i, option in enumerate(list_of_options):
            object_coord = list_of_options_3d_coord[i]
            displacement_world = object_coord - frame_tvec
            distance = np.linalg.norm(displacement_world).item()
            options_distances[option].append(distance)

    for option in list_of_options:
        relative_movement = get_relative_movement(options_distances[option])   # ! TODO need to feed this to the qwen prompt
        if relative_movement is None:
            del options_relative_movement[option]
        else:
            options_relative_movement[option] = relative_movement   # ! TODO need to ensure the length of options_relative_movement is > 1
    if len(options_relative_movement) <= 1:
        return {}
    
    list_of_options = list(options_relative_movement.keys())
    stop_frame_for_qwen = (df_Pid.iloc[action_indice[-1]+1]['start_frame'].item() - int(query_video_stop_frame)) // 3 + int(query_video_stop_frame)
    # print(f"stop_frame_for_qwen: {stop_frame_for_qwen}, query_video_stop_frame: {query_video_stop_frame}")
    question, answer = fill_furniture_affordance_prediction_template(
        narrations,
        list_of_options, 
        predictor, 
        groundtruth_anchor_object, 
        options_relative_movement, 
        query_video_start_frame, 
        stop_frame_for_qwen,
        rgb_main_dir=rgb_main_dir,
        video_id=video_id,
        uniform_sample_max_frames=uniform_sample_max_frames,
        rgb_mode=rgb_mode
    )
    return {
        # ! TODO
        "qa_type": "furniture_affordance",
        "template_type": "single",
        "query_start_frame": query_video_start_frame,
        "query_stop_frame": query_video_stop_frame,
        "question": question,
        "answer": answer
    }

def generate_action_planning_qa():
    pass

def get_small_object_3d_coord(tmp_data, 
                              tmp_data_ann, 
                              visor_to_epic_map, 
                              rgb_main_dir, 
                              rgb_mode, 
                              video_id, 
                              colmap_images, 
                              colmap_points3D, 
                              frame_index_to_colmap_imageid_dict):
    """
    Extract 3D coordinates of a small object from 2D segmentation mask.
    
    Args:
        tmp_data: Image data containing name and video information
        tmp_data_ann: Annotation data for the object
        visor_to_epic_map: Mapping from visor to epic image names
        rgb_main_dir: Main directory for RGB data
        rgb_mode: Mode (train/val/test)
        video_id: Video identifier
        colmap_images: COLMAP images data
        colmap_points3D: COLMAP 3D points data
        frame_index_to_colmap_imageid_dict: Mapping from frame index to COLMAP image ID
        
    Returns:
        object_3d_coord: 3D coordinates of the object
    """
    image_name = tmp_data['image']['name']
    scene_id = tmp_data['image']['video']
    epic_image_name = visor_to_epic_map[scene_id][image_name]
    epic_image_path = os.path.join(rgb_main_dir, rgb_mode, video_id.split('_')[0], scene_id, epic_image_name)
    seg_mask = generate_masks_for_image(image_name="tmp.jpg",
                            image_path=epic_image_path,
                            masks_info=[tmp_data_ann],
                            output_directory='./not_easily_movable_objects_masks',
                            output_resolution=(456,256))

    image_id, improvable = find_nearby_key(extract_index(epic_image_name), frame_index_to_colmap_imageid_dict)
    object_3d_coord = get_3d_coord_from_2d_segmentation(seg_mask, image_id, colmap_images, colmap_points3D)
    return object_3d_coord


def video_clip_sampling(qa_type, 
                        video_id, 
                        df_Pid, 
                        min_len=5, 
                        max_len=10, 
                        search_limit=100, 
                        rgb_mode='train', 
                        list_of_options=None, 
                        list_of_options_id=None,
                        segment_json=None
                        ):
    """
    Sample video clips for different QA types.
    
    Args:
        qa_type: Type of question-answer (relative_direction, relative_distance, find_my_item, furniture_affordance, action_planning)
        video_id: Video identifier
        df_Pid: DataFrame containing action labels
        min_len: Minimum length of video clip
        max_len: Maximum length of video clip
        search_limit: Maximum number of search attempts
        rgb_mode: Mode (train/val/test)
        list_of_options: List of option names
        list_of_options_id: List of option IDs
        segment_json: Segmentation annotations
        
    Returns:
        For relative_direction/relative_distance: list of action label row indices
        For find_my_item: (indices, object_name, tmp_data, tmp_data_ann)
        For furniture_affordance: (indices, anchor_object)
        For action_planning: indices (not implemented)
    """
    # output: a list of action label row index
    total_rows = len(df_Pid)
    if "relative_direction" in qa_type or "relative_distance" in qa_type:
        # random sample 5-10 actions
        chunk_len = random.randint(min_len, max_len)
        start_idx = random.randint(0, total_rows - chunk_len)
        indices = list(range(start_idx, start_idx + chunk_len))

    elif "find_my_item" in qa_type:
        count = 0
        indices = []
        video_anns = segment_json['video_annotations']
        
        while True:
            # get random item from the seg_json, exclude the anchor object, and also the right hand and left hand
            # ! TODO
            # check object name in start_idx noun 
            tmp_data = None
            tmp_data_ann = None
            for i in range(50):
                random_data = random.choice(video_anns)
                for random_data_ann in random_data['annotations']:
                    if random_data_ann['class_id'] in small_objects_id_to_name_dict.keys():
                        tmp_data = random_data
                        tmp_data_ann = random_data_ann
                        break
                if tmp_data is not None:
                    break
            object_id = random_data_ann['class_id']
            print(f'1181 utils.py: object_id: {object_id}')

            matching_indices = df_Pid[df_Pid['all_noun_classes'].apply(lambda x: object_id in ast.literal_eval(x))].index.tolist()
            print(f"Example row: {df_Pid['all_noun_classes'].iloc[0]}")
            print(f"object_id: {object_id}, type: {type(object_id)}")
            print(f'1184 utils.pymatching_indices : {matching_indices}')
            if matching_indices != []:
                start_idx = random.choice(matching_indices)
            else:
                continue

            chunk_len = random.randint(min_len, max_len)

            object_name = small_objects_id_to_name_dict[object_id]
            print(f'1193 utils.py: object_name: {object_name}')
            
            shift_attempt = 0
            found_the_chunk = False
            while shift_attempt < 10 and (start_idx + chunk_len < total_rows):
                conflict_found = False
                for i in range(chunk_len + 1):
                    
                    if df_Pid.iloc[start_idx + i]['noun'] == object_name:
                        conflict_found = True
                        print(f'1201 utils.py: conflict_found: {conflict_found}')
                        break

                
                if conflict_found:
                    print(f'1204 utils.py: conflict_found: {conflict_found}')
                    start_idx += 1
                    shift_attempt += 1
                else:
                    print(f'1207 utils.py: found the chunk with start_idx: {start_idx}, chunk_len: {chunk_len}')
                    found_the_chunk = True
                    break

            count += 1
            if count > search_limit or found_the_chunk:
                if found_the_chunk:
                    indices = list(range(start_idx, start_idx + chunk_len))
                    print(f'1214 utils.py found the chunk with indices: {indices}')
                else:
                    indices = []
                break
            if count > 25 and chunk_len > min_len:
                chunk_len = min_len

        return indices, object_name, tmp_data, tmp_data_ann
            
        
    elif "furniture_affordance" in qa_type:
        count = 0 
        while True:  # ! TODO ensure previous narration do not contain the anchor object
            object_id = random.choice(list_of_options_id)
            anchor_object = anchor_objects_id_to_name_dict[object_id]
            # matching_rows = df_Pid[df_Pid["narration"].str.contains(anchor_object, case=False, na=False)]
            matching_indice = df_Pid[df_Pid['all_noun_classes'].apply(lambda x: object_id in ast.literal_eval(x))].index.tolist()
            print(f'1240 utils.py matching_rows: {matching_indice}, anchor_object: {anchor_object}')
            if matching_indice != []:
                end_idx = random.choice(matching_indice)  # convert to integer
                chunk_len = random.randint(min_len, max_len)
                chunk_len = end_idx if end_idx - chunk_len < 0 else chunk_len  # remove +1
                start_idx = end_idx - chunk_len
                indices = list(range(start_idx, end_idx))  # no +1, exclude end_idx
                print(f'1246 utils.py found the chunk with start_idx: {start_idx}, end_idx: {end_idx}')
                break
            count += 1
            if count > search_limit:
                indices = []
                break
        return indices, anchor_object
    
    elif "action_planning" in qa_type:
        pass  # ! TODO add the action planning sampling
                
    return indices

def _generate_qa(qa_type):
    if "relative_direction" in qa_type:
        return generate_relative_direction_qa
    elif "relative_distance" in qa_type:
        return generate_relative_distance_qa
    elif "find_my_item" in qa_type: 
        return generate_find_my_item_qa
    elif "furniture_affordance" in qa_type:
        return generate_furniture_affordance_qa
    elif "action_planning" in qa_type:
        return generate_action_planning_qa   # ! Not implemented yet
    

def filter_dataframe_by_value(df, column_title, string_value):
    return df[df[column_title] == string_value]

def sort_dataframe_by_value(df, value):
    return df.sort_values(by=value, ascending=True).reset_index(drop=True)

def convert_numpy_to_python(obj):
    if isinstance(obj, dict):
        return {convert_numpy_to_python(k): convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def dict_to_frozenset(d):
    return frozenset(d.items())

    
def generate_qa(max_num=10000, 
                mode="train", 
                probabilities=[0.2, 0.2, 0.2, 0.2, 0.2], 
                output_dir=None, 
                output_filename=None,
                uniform_sample_max_frames = 10):
    """
    Generate question-answer pairs for spatio-temporal reasoning tasks.
    
    This function generates QA pairs for different types of spatial reasoning tasks:
    - relative_direction: Questions about directional relationships between an object and a person
    - relative_distance: Questions about distance relationships between an object and a person
    - find_my_item: Questions about locating specific items in the scene
    - furniture_affordance: Questions about what possible furniture can be interacted with
    - action_planning: Not implemented yet
    """
    
    with open(visor_to_epic_map_path, 'r') as f:
        visor_to_epic_map = json.load(f)

    mode = "train" if mode == "train" else "val"
    if mode == "train":
        df = pd.read_csv('/data/haozhen/4dllm/4D-LLM/epic-fields/annotations/EPIC_100_train.csv')
    else: 
        df = pd.read_csv('/data/haozhen/4dllm/4D-LLM/epic-fields/annotations/EPIC_100_validation.csv')

    with open(f"/data/haozhen/4dllm/4D-LLM/epic-fields/annotations/{mode}_video_ids.txt", 'r') as f:
        video_ids = [line.strip() for line in f]

    qa_types = ["relative_direction", "relative_distance", "find_my_item", "furniture_affordance", "action_planning"]

    # load qwen model

    predictor = VLMPredictor(model_name="Qwen/Qwen2-VL-7B-Instruct", device="cuda")

    # get num of scene_id and uniformly sample from each scene_id 
    count = 0
    final_output = []
    seen = set()
    pbar = tqdm(total=max_num)
    error_message = []
    while count < max_num:

        for video_id in video_ids:
            try:  # ! DEBUG
                # get reverse dict[epic_frame_index] = colmap_image_id
                epic_frame_index_to_colmap_key_json_path = os.path.join(colmap_main_dir, video_id, f"{mode}_frame_to_colmap_img_key_{video_id}_reset_index.json")
                with open(epic_frame_index_to_colmap_key_json_path, 'r') as f:
                    frame_index_to_colmap_imageid_dict = json.load(f)

                frame_index_to_colmap_imageid_dict = {int(k): v for k, v in frame_index_to_colmap_imageid_dict.items()}

                # get df_Pid: sorted epic kitchen label for video_id 
                df_Pid = filter_dataframe_by_value(df, "video_id", video_id)
                df_Pid = sort_dataframe_by_value(df_Pid, "start_frame")
                

                # get VISOR segment json path
                segment_json_path = os.path.join(visor_gt_json_main_dir, "train", video_id + '.json')
                if not os.path.exists(segment_json_path):
                    segment_json_path = os.path.join(visor_gt_json_main_dir, "test", video_id + '.json')
                    if not os.path.exists(segment_json_path):
                        continue    # ! TODO modify later if  no visor json file exist
                
                # read colmap model
                input_model_path = os.path.join(colmap_model_main_dir, video_id)
                colmap_cameras, colmap_images, colmap_points3D = read_model(input_model_path)


                # read VISOR segment annotation json 
                with open(segment_json_path, 'r') as f:
                    segment_json = json.load(f)
                
                # for-loop 
                #  caluclate anchor objects 3D coordinates
                not_easily_movable_objects_masks = generate_masks_for_all_anchor_objects(visor_to_epic_map=visor_to_epic_map, segment_json=segment_json, mode=mode, rgb_main_dir=rgb_main_dir, output_directory=None, frame_index_to_colmap_imageid_dict=frame_index_to_colmap_imageid_dict, colmap_images=colmap_images)
            
                # get a dict -> dict[obj_id] = average 3d coord for 2d mask 
                anchor_objects_3d_coords_dict = get_3d_coord_for_all_anchor_objects(not_easily_movable_objects_masks, colmap_images, colmap_points3D)

                # print(f'1434 utils.py anchor_objects_3d_coords_dict: {anchor_objects_3d_coords_dict}')
                # import pdb; pdb.set_trace() # ! DEBUG
                num_of_qa_generated_per_scene = max_num // len(video_ids) if max_num > len(video_ids) else max_num
                for _ in range(num_of_qa_generated_per_scene):
                    try: # ! DEBUG
                        qa_type = random.choices(qa_types, probabilities, k=1)[0]
                        if ("relative_direction" in qa_type or "relative_distance" in qa_type) and len(anchor_objects_3d_coords_dict) == 0:
                            continue
                        
                        if "furniture_affordance" in qa_type and len(anchor_objects_3d_coords_dict) < 2:
                            continue 
                        elif "furniture_affordance" in qa_type:
                            max_num_of_options = min(len(list(anchor_objects_3d_coords_dict.keys())), 4)
                            option_ids = random.sample(list(anchor_objects_3d_coords_dict.keys()), max_num_of_options)
                            list_of_options = [anchor_objects_id_to_name_dict[i] for i in option_ids]
                            action_indice, groundtruth_anchor_object = video_clip_sampling(qa_type, video_id, df_Pid, min_len=5, max_len=10, search_limit=100, list_of_options=list_of_options, list_of_options_id=option_ids)
                        elif "find_my_item" in qa_type:
                            action_indice, groundtruth_anchor_object, tmp_data, tmp_data_ann = video_clip_sampling(qa_type, video_id, df_Pid, min_len=5, max_len=10, search_limit=100, segment_json=segment_json)
                            print(f'1374 utils.py action_indice: {action_indice}')
                            if action_indice != []:
                                object_3d_coord = get_small_object_3d_coord(tmp_data, 
                                                                            tmp_data_ann, 
                                                                            visor_to_epic_map, 
                                                                            rgb_main_dir, 
                                                                            'test' if mode == 'val' else 'train', 
                                                                            video_id, 
                                                                            colmap_images, 
                                                                            colmap_points3D, 
                                                                            frame_index_to_colmap_imageid_dict)
                                print(f'1385 utils.py object_3d_coord: {object_3d_coord}')
                                if object_3d_coord is None:
                                    continue
                        else:
                            action_indice = video_clip_sampling(qa_type, video_id, df_Pid, min_len=5, max_len=10, search_limit=100)

                        
                        if action_indice == []:
                            print(f"action_indice is empty")
                            continue

                        if "relative_direction" in qa_type:
                            data_dict = generate_relative_direction_qa(action_indice=action_indice, 
                                                                df_Pid=df_Pid, 
                                                                frame_index_to_colmap_imageid_dict=frame_index_to_colmap_imageid_dict, 
                                                                colmap_images=colmap_images,
                                                                anchor_objects_3d_coords_dict=anchor_objects_3d_coords_dict)
                        elif "relative_distance" in qa_type:
                            data_dict = generate_relative_distance_qa(action_indice=action_indice,
                                                                df_Pid=df_Pid, 
                                                                frame_index_to_colmap_imageid_dict=frame_index_to_colmap_imageid_dict, 
                                                                colmap_images=colmap_images,
                                                                anchor_objects_3d_coords_dict=anchor_objects_3d_coords_dict)
                        elif "find_my_item" in qa_type: 
                            data_dict = generate_find_my_item_qa(action_indice=action_indice,
                                                            df_Pid=df_Pid, 
                                                            predictor=predictor,
                                                            frame_index_to_colmap_imageid_dict=frame_index_to_colmap_imageid_dict, 
                                                            colmap_images=colmap_images,
                                                            object_3d_coord=object_3d_coord,   # ! TODO need to add the 3d coord of the small object
                                                            rgb_main_dir=rgb_main_dir,
                                                            video_id=video_id,
                                                            rgb_mode='test' if mode == 'val' else 'train',
                                                            anchor_objects_3d_coords_dict=anchor_objects_3d_coords_dict,)
                        elif "furniture_affordance" in qa_type:
                            list_anchor_objects_3d_coords = [anchor_objects_3d_coords_dict[anchor_objects_name_to_id_dict[k]] for k in list_of_options]
                            data_dict = generate_furniture_affordance_qa(action_indice=action_indice,
                                                                        df_Pid=df_Pid, 
                                                                        frame_index_to_colmap_imageid_dict=frame_index_to_colmap_imageid_dict, 
                                                                        colmap_images=colmap_images,
                                                                        groundtruth_anchor_object=groundtruth_anchor_object,
                                                                        list_of_options_3d_coord=list_anchor_objects_3d_coords,
                                                                        list_of_options=list_of_options, 
                                                                        predictor=predictor,
                                                                        video_id=video_id,
                                                                        rgb_main_dir=rgb_main_dir,
                                                                        uniform_sample_max_frames=uniform_sample_max_frames,
                                                                        rgb_mode='test' if mode == 'val' else 'train')
                        elif "action_planning" in qa_type:
                            raise NotImplementedError("Action planning is not implemented yet")
                            data_dict = generate_action_planning_qa()

                        if data_dict != {}:
                            count += 1 
                            print(data_dict)
                            data_dict['video_id'] = video_id
                            data_dict = convert_numpy_to_python(data_dict)
                            frozen = dict_to_frozenset(data_dict)
                            if frozen not in seen:
                                seen.add(frozen)
                                final_output.append(data_dict)
                                pbar.update(1)
                            if count >= max_num:
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                if output_dir is None:
                                    output_dir = './output'
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                output_filename = output_filename if output_filename is not None else f"qa_{mode}_{max_num}_{timestamp}.json"
                                with open(os.path.join(output_dir, output_filename), 'w') as f:
                                    json.dump(final_output, f)
                                with open(os.path.join(output_dir, f"error_message_{timestamp}.json"), 'w') as f:
                                    json.dump(error_message, f)
                                return  
                                
                    except Exception as e:
                        print(f"INNER: Error generating QA for video {video_id}: {e}")
                        error_message.append(f"Error generating QA for video {video_id}: {e}")
                        continue # ! DEBUG
            except Exception as e:
                print(f"OUTER: Error processing video {video_id}: {e}")
                error_message.append(f"Error processing video {video_id}: {e}")
                continue # ! DEBUG
    # safe save results to json file
    with open("safe_save_qa.json", 'w') as f:
        json.dump(final_output, f)   
    with open("safe_save_error_message.json", 'w') as f:
        json.dump(error_message, f)
    pbar.close()             
        

# task order: relative_direction, relative_distance, find_my_item, furniture_affordance, action_planning
generate_qa(max_num=25, mode="val", probabilities=[0, 0, 1, 0, 0], output_dir=None, output_filename=None)