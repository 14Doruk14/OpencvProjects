import os
import cv2
from PIL import Image
import numpy as np
import json

# Ensure the masks directory exists
mask_dir = r"C:\Users\hp\Desktop\computer vision\gastric_image\masks\stage1"
os.makedirs(mask_dir, exist_ok=True)

# Load JSON annotation file
with open(r"C:\Users\hp\Desktop\computer vision\gastric_image\stage1.json", "r") as read_file:
    data = json.load(read_file)

# List of all annotated file names
all_file_names = list(data.keys())

# Get files in the stage1 image directory
image_dir = r"C:\Users\hp\Desktop\computer vision\gastric_image\stage1"
files_in_directory = set(os.listdir(image_dir))  # Convert to set for faster lookup

for j in range(len(all_file_names)):
    image_name = data[all_file_names[j]]['filename']

    if image_name not in files_in_directory:
        continue  # Skip if the image file is missing

    # Load image
    img_path = os.path.join(image_dir, image_name)
    img = np.asarray(Image.open(img_path))

    # Process regions if available
    if data[all_file_names[j]]['regions']:
        try:
            shape1_x = data[all_file_names[j]]['regions']['0']['shape_attributes']['all_points_x']
            shape1_y = data[all_file_names[j]]['regions']['0']['shape_attributes']['all_points_y']
        except KeyError:
            shape1_x = data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_x']
            shape1_y = data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_y']

        # Stack points and create mask
        polygon = np.stack((shape1_x, shape1_y), axis=1)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(int)], 255)  # Fill the mask with white

        # Generate the output mask filename
        mask_filename = os.path.splitext(image_name)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_filename)

        # Save the mask
        cv2.imwrite(mask_path, mask)

        print(f"Saved: {mask_path}")
