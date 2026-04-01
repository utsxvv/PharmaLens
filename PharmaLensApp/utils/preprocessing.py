import cv2
import numpy as np

def preprocess_image(filepath):

    img = cv2.imread(filepath)

    # Resize
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur (noise removal)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh