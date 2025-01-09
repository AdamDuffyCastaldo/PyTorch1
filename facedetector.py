import cv2
import torch
import numpy as np
from PIL import Image
import dlib


def Load_image(path):
    img = cv2.imread(path)
    return img

path = input("Enter the path to your image: ")
img = Load_image(path)



detector = dlib.get_frontal_face_detector()
prediction = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")



grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rectangle = detector(grayscale, 0)
if len(rectangle) > 0:
    for rect in rectangle:
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()
        shape = prediction(grayscale, rect)

        for i in range(shape.num_parts):
            part = shape.part(i)
            cv2.circle(img, (part.x, part.y), 3, (0, 255, 0), -1)  # Green circle
        
        # Draw bounding box
        cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)  # Blue rectangle

        print(shape)


def shape_normal(shape):
    shape_normal = []
    for i in range(5):
        shape_normal.append((i, shape.part(i).x, shape.part(i).y))
    return shape_normal

def eyesandnose_dlib(shape):
    nose = shape[4][1], shape[4][2]
    left_eye_x = int(shape[3][1] + shape[2][1]) // 2
    left_eye_y = int(shape[3][2] + shape[2][2]) // 2
    right_eyes_x = int(shape[1][1] + shape[0][1]) // 2
    right_eyes_y = int(shape[1][2] + shape[0][2]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

shape = shape_normal(shape)
nose, l_eye, r_eye = eyesandnose_dlib(shape)
centre = ((l_eye[0]+r_eye[0])//2, (l_eye[1]+r_eye[1])//2)

centre_prediction = (int((x + w) / 2), int((y + y) / 2))


def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

length1 = dist(centre, nose)
length2 = dist(centre_prediction, nose)
length3 = dist(centre_prediction, centre)

def cosrule(length1, length2, length3):
    cos = -(length3**2 - length2**2 - length1**2) / (2*length2 * length1)
    return cos


cos1 = cosrule(length1, length2, length3)
angle = np.arccos(cos1)

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle)*(px - ox) - np.sin(angle)*(py - oy)
    qy = oy + np.sin(angle)*(px - ox) + np.cos(angle)*(py - oy)
    return qx, qy

def inside(A, B, C, P):
    CrossProduct1 = (B[0]-A[0])*(P[1]-A[1]) - (B[1] - A[1])*(P[0]-A[0])
    CrossProduct2 = (C[0]-B[0])*(P[1]-B[1]) - (C[1] - B[1])*(P[0]-B[0])
    CrossProduct3 = (A[0]-C[0])*(P[1]-C[1]) - (A[1] - C[1])*(P[0]-C[0])

    if (CrossProduct1 < 0 and CrossProduct2 < 0 and CrossProduct3 < 0) or (CrossProduct1 > 0 and CrossProduct2 > 0 and CrossProduct3 > 0):
        return True
    else:
        return False

new_rotated_point = rotate(nose, centre, angle)
new_rotated_point = (int(new_rotated_point[0]), int(new_rotated_point[1]))
if inside(nose, centre, centre_prediction, new_rotated_point):
    angle = np.degrees(-angle)
else:
    angle = np.degrees(angle)



img = Image.fromarray(img)
img = np.array(img.rotate(angle))
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()