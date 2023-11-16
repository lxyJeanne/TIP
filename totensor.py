from PIL import Image
import os
import torch
from torchvision import transforms
import csv

class ImageData:
    def __init__(self, y, x):
        self.x = x
        self.y = y


def load_labels(csv_path):
    # Initialize an empty dictionary to store labels
    labels = {}

    with open(csv_path, 'r') as csvfile:
        # Use csv.reader to read the CSV file
        csv_reader = csv.reader(csvfile)

        # Skip the header
        next(csv_reader)

        # Process each row of data
        for row in csv_reader:
            # Extract filename and label from the row
            filename, label = row[0] + ".jpg", row[1]

            # Add filename and label to the dictionary
            labels[filename] = label

    # Return a dictionary containing filenames and labels
    return labels


def process_images(folder_path, labels_csv):
    # Get the filenames of all images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Load label data
    image_labels = load_labels(labels_csv)
    image_data_list = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        label = image_labels.get(image_file, 'Unknown')
        # Open the image using Pillow
        image = Image.open(image_path)

        # Define a transformation to convert the image to a PyTorch tensor
        transform = transforms.ToTensor()

        # Apply the transformation to the image
        tensor_image = transform(image)

        image_data=ImageData(y=label, x=tensor_image)
        image_data_list.append(image_data)

    return image_data_list


# Example usage
folder_path = 'train-resizedtest'
labels_csv = 'train-labels.csv'

image_data_list =process_images(folder_path, labels_csv)
for image_data in image_data_list:
    print("Label:", image_data.y)
    print("Image Tensor Shape:", image_data.x.shape)


