import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import cv2
import torchvision.models.segmentation as segmentation

model = segmentation.deeplabv3_resnet101(weights=segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
model.eval()

def load_input_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_image, input_batch

def segment_image(image_path):
    input_image, input_batch = load_input_image(image_path)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    mask = output_predictions == 15
    background = input_image.convert("L")
    background = background.filter(ImageFilter.DETAIL)
    segmented_image = np.array(input_image)
    background_array = np.array(background)
    for c in range(3):
        segmented_image[:, :, c] = np.where(mask, segmented_image[:, :, c], background_array)
    return Image.fromarray(segmented_image)

image_path = '../ImageDirectory/Lenna.png'
segmented_img = segment_image(image_path)

segmented_img_cv = np.array(segmented_img)
segmented_img_cv = segmented_img_cv[:, :, ::-1].copy()

cv2.imshow('Segmented Image', segmented_img_cv)
cv2.waitKey(0)
