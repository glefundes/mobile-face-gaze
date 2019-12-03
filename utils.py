import cv2
import json
import torch
import numpy as np

def draw_gaze(image_in, eye_pos, pitchyaw, length=200, thickness=1, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.5)
    return image_out

def convert_to_unit_vector(angles):
    x = -torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = -torch.sin(angles[:, 0])
    z = -torch.cos(angles[:, 1]) * torch.cos(angles[:, 1])
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(preds, labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


def normalize_face(landmarks, frame):
    # Adapted from imutils package
    left_eye_coord=(0.70, 0.35)
    
    lcenter = tuple([landmarks[0],landmarks[5]])
    rcenter = tuple([landmarks[1],landmarks[6]])
    
    gaze_origin = (int((lcenter[0]+rcenter[0])/2), int((lcenter[1]+rcenter[1])/2))

    # compute the angle between the eye centroids 
    dY = rcenter[1] - lcenter[1]
    dX = rcenter[0] - lcenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    
    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    right_eye_x = 1.0 - left_eye_coord[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    new_dist = (right_eye_x - left_eye_coord[0])
    new_dist *= 112
    scale = new_dist / dist

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(gaze_origin, angle, scale)

    # update the translation component of the matrix
    tX = 112 * 0.5
    tY = 112 * left_eye_coord[1]
    M[0, 2] += (tX - gaze_origin[0])
    M[1, 2] += (tY - gaze_origin[1])

    # apply the affine transformation
    face = cv2.warpAffine(frame, M, (112, 112),
        flags=cv2.INTER_CUBIC)
    return face, gaze_origin, M