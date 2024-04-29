# 1. vintage_filter
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/f1eceea1-8ae9-49c3-987a-b9ee27d52063)
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/51e4167e-a85a-4afc-8ac5-dac6fc8b81f2)
## OpenCV 라이브러리와 NumPy를 사용하여 구현
## 1. 세피아 필터 적용: 주어진 이미지에 세피아 컬러 변환을 적용
## 2. 가우시안 노이즈 추가: 변환된 이미지에 가우시안 노이즈를 추가하여 오래된 사진의 질감을 표현
## 3. 마스크 생성 및 블러 처리: 이미지 중심에 원형 마스크를 생성하고 가우시안 블러를 적용하여 중심이 뚜렷하고 가장자리가 흐릿한 효과 추가
## 4. 빈티지 효과 적용: 노이즈가 추가된 세피아 이미지와 블러 처리된 마스크를 합성하여 최종적으로 빈티지 효과를 적용한 이미지를 생성
# 2. dreamy_filter
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/066c647b-deed-43c1-b277-d2c16024a7da)
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/1a7a53b8-b749-4b96-a6db-5ddaabd53ed3)
## OpenCV 라이브러리와 NumPy를 사용하여 구현
## 1. 블러 효과 적용: 입력 이미지에 가우시안 블러(Gaussian Blur)를 적용해 부드러운 윤곽 표현
## 2. 색상 강조: 블러 처리된 이미지를 HSV 색공간으로 변환한 후, 채도(Saturation)와 명도(Value)를 증가시켜 색상을 더 밝고 선명하게 함
## 3. light leak 효과 적용: light_leak 함수를 통해 이미지의 특정 부분에 색상을 추가, 이를 통해 이미지에 빛 새는 듯한 효과를 부여함 (우측 상단)
# 3. cartoon_filter
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/6b0a3c2e-3d71-49fa-bb75-b53d22c6d361)
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/b77ecfc6-081e-40f7-b0f4-42f97f48d387)
## OpenCV와 NumPy 라이브러리를 사용하여 구현
## 1. 색상 양자화: cv2.kmeans를 사용하여 이미지의 색상을 K개의 대표 색상으로 양자화 (이미지의 색상 범위를 줄여 만화 같은 효과)
## 2. 에지 검출: 입력 이미지를 그레이스케일로 변환 후, cv2.medianBlur로 노이즈를 줄이고, cv2.adaptiveThreshold를 사용하여 이미지의 에지 검출
## 3. 만화 효과 적용: 양자화된 색상 이미지와 에지 검출 이미지를 결합하여 만화 효과를 적용 (cv2.bitwise_and 함수를 사용하여 양자화된 색상 이미지에 에지 검출 이미지를 마스크로 적용)
# 4. face_warping_filter
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/474f18b3-5b68-4995-9ff8-06dd437a827c)
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/c371dbec-b4e3-4908-9f97-5bd45896fe16)
## 1. cv2, dlib, numpy 라이브러리를 사용하여 구현
## 2. load_landmarks 함수: 이미지에서 얼굴을 검출하고, 그 얼굴에 대한 랜드마크를 반환
## 3. cv2.cvtColor(image, cv2.COLOR_BGR2GRAY): 이미지를 흑백으로 변환하여 얼굴 검출 능력 향상, (detector(gray): 흑백 이미지에서 얼굴을 검출), predictor(gray, faces[0]): 검출된 첫 번째 얼굴에 대해 랜드마크를 검출
## 4. enlarge_eyes 함수: 이미지 경로와 확대 비율(scale)을 입력받아, 해당 이미지에서 눈을 확대 [dlib.get_frontal_face_detector(): 얼굴 검출기를 초기화, dlib.shape_predictor('shape_predictor_68_face_landmarks.dat'): 랜드마크 예측기를 초기화, load_landmarks(image, predictor, detector): 위에서 정의한 함수를 사용하여 얼굴 랜드마크를 로드, 눈의 랜드마크(왼쪽 눈은 36:41, 오른쪽 눈은 42:47)를 사용하여 각 눈의 중심을 계산, 해당 중심을 기준으로 눈 영역(ROI)을 확대, 확대된 눈 영역을 원본 이미지에 다시 삽입]
## 5. cv2.imshow("Enlarged Eyes", image): 확대된 눈이 적용된 이미지를 나타냄
# 5. custom_filter
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/f1eceea1-8ae9-49c3-987a-b9ee27d52063)
![image](https://github.com/Akaps7777/MachineVision/assets/145246610/1cac38b4-a9ee-4243-ac86-6c6f93161c65)
## PyTorch와 torchvision의 CNN 모델, PIL, NumPy, OpenCV 라이브러리를 사용하여 구현
## 딥러닝 기반의 CNN 모델을 활용, 주어진 이미지에서 특정 객체를 분리
## 1. 모델 load: Pre-trained된 DeepLabV3 모델(ResNet101을 feature extractor으로 사용)을 load (이미지 내의 다양한 객체를 분할하는 데 사용)
## 2. 이미지 전처리: 주어진 이미지를 로드하고, transforms 라이브러리를 사용하여 모델에 필요한 형태로 전처리 (텐서로 변환, 정규화가 포함됨)
## 3. 이미지 분할: 전처리된 이미지 배치를 모델에 입력하여 학습, 모델의 출력에서 각 픽셀의 클래스를 예측, 특정 클래스(예: 얼굴)에 해당하는 마스크를 생성
## 4. 배경 처리: 입력 이미지의 배경을 회색조로 변환, ImageFilter.DETAIL 필터를 적용하여 디테일을 강조
## 5. 분할 이미지 생성: 생성된 마스크를 사용하여 분할된 객체는 원본 색상 유지, 배경은 처리된 배경 이미지로 대체 (NumPy 사용)
## 6. 결과 출력: 최종 분할 이미지를 OpenCV를 사용하여 화면에 표시 (PIL 이미지를 OpenCV 형식(BGR)으로 변환)

