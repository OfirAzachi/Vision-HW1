import torch
import yaml
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.image as mpimg
import os
import shutil
from ultralytics import YOLO
from tqdm import tqdm

# Data Organization
def copy_files(src, dest):
    os.makedirs(dest, exist_ok=True)

    files = os.listdir(src)

    for file in files:
        source_file = os.path.join(src, file)
        destination_file = os.path.join(dest, file)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)

print('Copying labeled data.')
source_dir = '/home/student/Desktop/OfirVisionHW1/Data/labeled_image_data/images/train'
destination_dir = '/home/student/Desktop/OfirVisionHW1/TempDataHolders/images'
copy_files(source_dir, destination_dir)

source_dir = '/home/student/Desktop/OfirVisionHW1/Data/labeled_image_data/labels/train'
destination_dir = '/home/student/Desktop/OfirVisionHW1/TempDataHolders/labels'
copy_files(source_dir, destination_dir)


def get_pics_from_vid(vid_path, save_count, out_path='TempDataHolders/pseudo_images'):
    os.makedirs(out_path, exist_ok=True)
    cap = cv2.VideoCapture(vid_path)
    print('Extracting frames from vid.')
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    frame_interval = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(out_path, f'frame_{save_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            save_count += 1

        frame_count += 1

    cap.release()
    print('Finished!')
    return frame_count


save_count = 0
videos_path = ['/home/student/Desktop/OfirVisionHW1/Data/id_video_data/20_2_24_1.mp4',
               '/home/student/Desktop/OfirVisionHW1/Data/id_video_data/4_2_24_B_2.mp4',
               '/home/student/Desktop/OfirVisionHW1/Data/ood_video_data/4_2_24_A_1.mp4',
               '/home/student/Desktop/OfirVisionHW1/Data/ood_video_data/surg_1.mp4']


for pth in videos_path:
    save_count = get_pics_from_vid(pth, save_count)



# First Phase Training
model = YOLO("yolov8n.pt")

data = {
    'train': '/home/student/Desktop/OfirVisionHW1/Data/labeled_image_data/images/train',
    'val': '/home/student/Desktop/OfirVisionHW1/Data/labeled_image_data/images/val',
    'nc': 3,
    'names': ['Empty', 'Tweezers', 'Needle_driver']
}

with open('/home/student/Desktop/OfirVisionHW1/pic_data.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)


results = model.train(data='/home/student/Desktop/OfirVisionHW1/pic_data.yaml', epochs=100, imgsz=640, batch=4, warmup_epochs=5, degrees=8, shear=4, fliplr=0.3, translate=0.1)
os.makedirs(f'models', exist_ok=True)
model.save(f'/home/student/Desktop/OfirVisionHW1/models/model_base.pt')


for i in range(2):
    data = {
        'train': '/home/student/Desktop/OfirVisionHW1/TempDataHolders/images',
        'val': '/home/student/Desktop/OfirVisionHW1/Data/labeled_image_data/images/val',
        'nc': 3,
        'names': ['Empty', 'Tweezers', 'Needle_driver']
    }

    with open('/home/student/Desktop/OfirVisionHW1/pic_data.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    image_dir = '/home/student/Desktop/OfirVisionHW1/TempDataHolders/pseudo_images'

    predictions_dir = '/home/student/Desktop/OfirVisionHW1/TempDataHolders/labels'
    os.makedirs(predictions_dir, exist_ok=True)

    jump_cnt = 0
    for image_name in tqdm(os.listdir(image_dir)):
        if jump_cnt <= 0:
            image_path = os.path.join(image_dir, image_name)
            result = model.predict(source=image_path, verbose=False)[0]
            boxes = []
            confidences = []
            sum = 0
            lines = []

            for detection in result.boxes:
                class_id = int(detection.cls)
                lines.append(' '.join(map(str, [class_id] + [item.item() for item in detection.xywhn[0].to('cpu')])))

                sum += detection.conf[0]
            if len(result.boxes) != 0:
                if sum/len(result.boxes) >= 0.8:
                    with open(f'{predictions_dir}/{os.path.split(result.path)[-1]}'.replace('jpg', 'txt'), 'w') as file:
                        for line in lines:
                            file.write(line + '\n')
                    jump_cnt = 6

        jump_cnt -= 1

        source_file = f'{image_dir}/{os.path.split(result.path)[-1]}'
        destination_dir = predictions_dir.replace('labels', 'images')
        destination_file = os.path.join(destination_dir, os.path.basename(source_file))
        shutil.copy(source_file, destination_file)

    model = YOLO("yolov8n.pt")
    results = model.train(data='/home/student/Desktop/OfirVisionHW1/pic_data.yaml', epochs=100, imgsz=640, batch=4, warmup_epochs=5, degrees=8, shear=4, fliplr=0.3, translate=0.1)

    model.save(f'/home/student/Desktop/OfirVisionHW1/models/model_{i}.pt')

