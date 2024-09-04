import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import os
import torchvision.models as models
import numpy as np
import subprocess
import pandas as pd
import re
from torchvision.ops import roi_align


def load_vit_model():
    vit_model = models.vit_b_16(pretrained=True)
    vit_model.eval()
    return vit_model


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_transform():
    return T.Compose([
        T.ToTensor()
    ])


def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    hook = model.backbone.body.layer4.register_forward_hook(hook_fn)
    return model


def hook_fn(module, input, output):
    global feature_map
    feature_map = output

def detect_and_crop_objects(image_path, model, output_dir, threshold=0.8):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    vit_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image_tensor = transform(image)

    with torch.no_grad():
        predictions = model([image_tensor])
    features = feature_map
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in predictions[0]['labels'].numpy()]
    pred_boxes = predictions[0]['boxes'].detach().numpy()
    pred_scores = predictions[0]['scores'].detach().numpy()

    info_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_info.txt")
    with open(info_file_path, 'w') as f_info:
        for i, box in enumerate(pred_boxes):
            agent_flag = False
            if pred_scores[i] >= threshold:
                class_name = pred_classes[i]
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1

                f_info.write(
                    f"Object {i}: Class={class_name}, x1={x1}, y1={y1}, x2={x2}, y2={y2}, width={width}, height={height}\n")

                cropped_img = image.crop((x1, y1, x2, y2))
                # cropped_img_tensor = vit_transform(cropped_img).unsqueeze(0)

                roi = [torch.cat(
                    [torch.tensor([0]), pred_boxes[i].unsqueeze(0)])]

                pooled_features = roi_align(feature_map, [roi], output_size=(7, 7), spatial_scale=1.0)
                pooled_features = pooled_features.numpy()
                feature_file_path = os.path.join(output_dir, f"{class_name}_{i}_features.npy")
                np.save(feature_file_path, pooled_features)

                if class_name == 'person':
                    cropped_img_path = os.path.join(output_dir, f"person_{i}.jpg")
                    cropped_img.save(cropped_img_path)
                    if not agent_flag:
                        cropped_faces = run_openface(cropped_img_path, output_dir, i)
                    if len(cropped_faces) != 0:
                        agent_flag = True
                else:
                    cropped_img.save(os.path.join(output_dir, f"{class_name}_{i}.jpg"))


def run_openface(image_path, output_dir, openface_dir, crop_faces=True):

    os.makedirs(output_dir, exist_ok=True)

    landmark_executable = os.path.join(openface_dir, 'build/bin/FaceLandmarkImg')

    csv_output_path = os.path.join(output_dir, 'processed.csv')

    subprocess.run([
        landmark_executable,
        '-f', image_path,
        '-out_dir', output_dir,
        '-of', csv_output_path
    ], check=True)

    cropped_faces = []

    if os.path.exists(csv_output_path):
        df = pd.read_csv(csv_output_path)

        if not df.empty and ' success' in df.columns and df[' success'].iloc[0] == 1:
            image = Image.open(image_path)

            for i, row in df.iterrows():
                x = int(row[' face_x'])
                y = int(row[' face_y'])
                w = int(row[' face_width'])
                h = int(row[' face_height'])

                cropped_face = image.crop((x, y, x + w, y + h))

                cropped_face_filename = os.path.join(output_dir, f'face_{i}.jpg')
                cropped_face.save(cropped_face_filename)
                cropped_faces.append(cropped_face_filename)

    return cropped_faces



