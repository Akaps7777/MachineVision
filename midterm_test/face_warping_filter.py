import cv2
import dlib
import numpy as np
image = cv2.imread('../ImageDirectory/Lenna.png', cv2.IMREAD_COLOR)
if image is None: raise Exception('Image cannot be read')
def load_landmarks(image, predictor, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        return np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)
    return None

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

landmarks = load_landmarks(image, predictor, detector)

if landmarks is not None:
    for point in landmarks:
        cv2.circle(image, (point[0], point[1]), 1, (0, 255, 0), -1)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
