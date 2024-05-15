import mediapipe as mp
import numpy as np
import cv2
# import image_measure

def segment_body(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp.solutions.selfie_segmentation.SelfieSegmentation() as segmenter:
        results = segmenter.process(image_rgb)
        segmented_mask = (results.segmentation_mask * 255).astype(np.uint8)
        segmented_image = cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2BGR)
        return segmented_image

def get_landmark(path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    result = pose.process(image_rgb)

    landmark_px = []
    if result.pose_landmarks:
        height, width, _ = image.shape

        for landmark in result.pose_landmarks.landmark:
            x_px = int(landmark.x * width)
            y_px = int(landmark.y * height)
            if x_px <= width and y_px <= height:
                landmark_px.append((x_px, y_px))
        mp_drawing.draw_landmarks(
                image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image, landmark_px