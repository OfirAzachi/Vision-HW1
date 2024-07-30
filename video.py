from tqdm import tqdm
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


def draw_and_save_bounding_boxes(predictions, output_dir, class_names):
    for pred in tqdm(predictions):
        boxes = []
        confidences = []
        for detection in pred.boxes:
            class_id = int(detection.cls)
            boxes.append([class_id] + list(detection.xywhn[0].to('cpu')))
            confidences.append(detection.conf[0])

        img = cv2.imread(pred.path)
        for box in boxes:
            class_id, x_center, y_center, width, height = box

            img_height, img_width, _ = img.shape
            x_center = int(x_center * img_width)
            y_center = int(y_center * img_height)
            width = int(width * img_width)
            height = int(height * img_height)

            x_min = x_center - width // 2
            y_min = y_center - height // 2

            cv2.rectangle(img, (x_min, y_min), (x_min + width, y_min + height), (0, 0, 255), 2)

            class_label = class_names[int(class_id)]
            cv2.putText(img, class_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(f'{output_dir}_preds/{os.path.split(pred.path)[-1]}', img)


def extract_vid(output_dir, video_path):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    save_count = 0
    frame_interval = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{save_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            save_count += 1
        frame_count += 1

    cap.release()

    print("Finished extracting frames.")


def tag_video(output_dir, video_path):
    predictions_dir = f'{output_dir}_preds'
    os.makedirs(predictions_dir, exist_ok=True)

    results = model.predict(source=output_dir)

    draw_and_save_bounding_boxes(results, output_dir, ['Empty', 'Tweezers', 'Needle_driver'])
    print("Finished processing all images.")

    output_video_path = f'output_video_{output_dir}.mp4'

    images = sorted([img for img in os.listdir(predictions_dir)])

    first_image_path = os.path.join(predictions_dir, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

    for image_name in images:
        image_path = os.path.join(predictions_dir, image_name)
        img = cv2.imread(image_path)
        video_writer.write(img)
        print(f"Added {image_name} to video")

    video_writer.release()

    print(f"Video saved at {output_video_path}")

# Enter The Video url
video_path = ['/Data/ood_video_data/surg_1.mp4']
# Enter The Video Name
output_dir = ['surg_1']
for vid_pth, out_dir in zip(video_path, output_dir):
    extract_vid(out_dir, vid_pth)
    #Change the model name
    model = YOLO(f'./model.pt')
    tag_video(out_dir, vid_pth)