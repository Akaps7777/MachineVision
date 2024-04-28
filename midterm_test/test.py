import torch
from torchvision import models, transforms
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

# 사전 학습된 DeepLabV3 모델 불러오기
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()


# 이미지를 로드하고 전처리하기
def load_input_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # 배치 차원을 추가
    return input_image, input_batch


# 이미지에서 사람을 구분하여 배경을 흑백으로 처리하기
def segment_image(image_path):
    input_image, input_batch = load_input_image(image_path)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # 사람에 해당하는 부분은 15번 라벨입니다.
    mask = output_predictions == 15

    # 배경을 흑백으로 변환
    background = input_image.convert("L")
    background = background.filter(ImageFilter.DETAIL)

    # 컬러 이미지와 흑백 배경 이미지를 합치기
    segmented_image = np.array(input_image)
    background_array = np.array(background)
    for c in range(3):  # RGB 채널을 반복
        segmented_image[:, :, c] = np.where(mask, segmented_image[:, :, c], background_array)

    return Image.fromarray(segmented_image)


# 이미지 세그멘테이션 실행
image_path = '../ImageDirectory/Lenna.png'  # 이미지 경로 지정
segmented_img = segment_image(image_path)

# 결과 이미지 보여주기
plt.imshow(segmented_img)
plt.axis('off')
plt.show()
