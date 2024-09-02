import os
from PIL import Image
import re
import pandas as pd


def apply_mask_to_image(mask_path, image_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # resize mask
    mask = mask.resize(image.size, Image.NEAREST)

    # apply mask to image
    masked_image = Image.composite(image, Image.new("RGB", image.size), mask)

    # return image obj
    return masked_image
    # print(f"Masked image saved to: {output_image_path}")


def categorize_files(folder_path):
    image_files = []
    npy_files = []
    txt_files = []

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(image_extensions):
                image_files.append(file_path)
            elif file.lower().endswith('.npy'):
                npy_files.append(file_path)
            elif file.lower().endswith('.txt'):
                txt_files.append(file_path)

    return image_files, npy_files, txt_files


def read_object_info_from_txt(file_path):
    objects_info = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()  # 去除行首尾的空格和换行符
            if line.startswith("Object"):
                # 解析行内容
                parts = line.split(', ')

                obj_info = {}
                obj_info['object_id'] = int(parts[0].split(' ')[1].replace(':', ''))
                obj_info['class'] = parts[1].split('=')[1]
                obj_info['x1'] = float(parts[2].split('=')[1])
                obj_info['y1'] = float(parts[3].split('=')[1])
                obj_info['x2'] = float(parts[4].split('=')[1])
                obj_info['y2'] = float(parts[5].split('=')[1])
                obj_info['width'] = float(parts[6].split('=')[1])
                obj_info['height'] = float(parts[7].split('=')[1])

                objects_info.append(obj_info)

    return objects_info


def find_face_image_index(folder_path):
    face_indices = []

    pattern = re.compile(r'face_(\d+)\.jpg', re.IGNORECASE)

    for root, _, files in os.walk(folder_path):
        for file in files:
            match = pattern.match(file)
            if match:
                index = int(match.group(1))
                face_indices.append(index)
                # face_image = Image.open(os.path.join(folder_path, f'face_{index}.jpg'))

    if face_indices:
        return face_indices[0]
    else:
        return -1


from PIL import Image


def resize_depth_map_to_match_image(depth_map_path, original_image_path):
    original_image = Image.open(original_image_path)
    depth_map = Image.open(depth_map_path)

    original_size = original_image.size

    resized_depth_map = depth_map.resize(original_size, Image.ANTIALIAS)

    return resized_depth_map


def get_object_center_by_objs(objects_info, object_index):
    return ((objects_info[object_index].x1 + objects_info[object_index].x2) / 2,
            (objects_info[object_index].y1 + objects_info[object_index].y2) / 2)


def get_object_center(object_info):
    return ((object_info.x1 + object_info.x2) / 2,
            (object_info.y1 + object_info.y2) / 2)


def read_gaze_angles(csv_file_path):
    df = pd.read_csv(csv_file_path)

    if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns:
        gaze_angle_x = df['gaze_angle_x']
        gaze_angle_y = df['gaze_angle_y']
        return gaze_angle_x, gaze_angle_y
    else:
        print("Can not find required column!")
        return None, None


def apply_mask_to_specific_area(original_image_path, mask_image_path, area):

    original_image = Image.open(original_image_path).convert("RGB")
    mask_image = Image.open(mask_image_path).convert("L")

    mask_image = mask_image.resize((area[2], area[3]), resample=Image.BILINEAR)

    cropped_original = original_image.crop((area[0], area[1], area[0] + area[2], area[1] + area[3]))

    masked_cropped = Image.composite(cropped_original, Image.new("RGB", cropped_original.size, (0, 0, 0)), mask_image)

    original_image.paste(masked_cropped, (area[0], area[1]), mask_image)

    return original_image