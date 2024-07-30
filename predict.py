import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
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

def draw_bounding_boxes(img, boxes, classes=None):
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in boxes:
        class_id, x_center, y_center, width, height = box

        img_height, img_width, _ = img.shape
        x_center = x_center * img_width
        y_center = y_center * img_height
        width = width * img_width
        height = height * img_height

        x_min = x_center - width / 2
        y_min = y_center - height / 2

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

        if classes:
            class_label = classes[int(class_id)]
            plt.text(x_min, y_min, class_label, color='white', verticalalignment='top', bbox={'color': 'red', 'pad': 0})

    plt.axis('off')
    plt.show()

print(f'Segmenting photo:')
# Enter Image Path Below!!!
image_pth = ''
model = YOLO('./model.pt')
results = model.predict(source=image_pth, save=True, save_dir='./photo_results')
result = results[0]
boxes = []
confidences = []
print(f"Results for {result.path}:")
for detection in result.boxes:
    class_id = int(detection.cls)
    boxes.append([class_id] + list(detection.xywhn[0].to('cpu')))
    confidences.append(detection.conf[0])
img = mpimg.imread(result.path)

classes = {0: 'Empty', 1: 'Tweezers', 2: 'Needle_driver'}

draw_bounding_boxes(img, boxes, classes)
print(confidences)
