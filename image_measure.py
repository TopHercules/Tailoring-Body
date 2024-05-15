import cv2

def measure_height(segmented_image):
    if segmented_image is None or segmented_image.size == 0:
        print("Error: Segmented image is empty or None!")
        return 0
    try:
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"Error converting image to grayscale: {e}")
        return 0
    try:
        contours, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error as e:
        print(f"Error finding contours: {e}")
        return 0
    if not contours:
        return 0
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    _, _, _, h = cv2.boundingRect(contour)
    return h