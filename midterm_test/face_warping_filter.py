import cv2
import dlib
import numpy as np
def load_landmarks(image, predictor, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        return np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)
    return None

def enlarge_eyes(image_path, scale=1.5):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    image = cv2.imread(image_path)
    landmarks = load_landmarks(image, predictor, detector)
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    for eye in [left_eye, right_eye]:
        center = eye.mean(axis=0).astype("int")
        x, y = center
        width = int(np.max(eye[:, 0]) - np.min(eye[:, 0])) * scale
        height = int(np.max(eye[:, 1]) - np.min(eye[:, 1])) * scale
        new_x = int(x - width / 2)
        new_y = int(y - height / 2)

        eye_roi = image[new_y:new_y + int(height), new_x:new_x + int(width)]
        eye_roi_resized = cv2.resize(eye_roi, (0, 0), fx=scale, fy=scale)

        image[new_y:new_y + eye_roi_resized.shape[0], new_x:new_x + eye_roi_resized.shape[1]] = eye_roi_resized

    cv2.imshow("Enlarged Eyes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

enlarge_eyes("../ImageDirectory/Lenna.png")
