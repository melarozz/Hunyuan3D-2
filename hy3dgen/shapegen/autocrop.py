import cv2
import numpy as np
from PIL import Image


def crop_plant(image_path, lower_green=np.array([25, 40, 40]), upper_green=np.array([85, 255, 255]),
               margin_x_percent=0.1, margin_y_percent=0.3):
    """
    Loads image from image_path, looks for plant by green colour,
    adds margin and return cropped image in PIL Image format.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        margin_x = int(margin_x_percent * w)
        margin_y = int(margin_y_percent * h)
        x_new = max(x - margin_x, 0)
        y_new = max(y - margin_y, 0)
        w_new = min(w + 2 * margin_x, img.shape[1] - x_new)
        h_new = min(h + 2 * margin_y, img.shape[0] - y_new)
        cropped = img[y_new:y_new + h_new, x_new:x_new + w_new]
    else:
        # if couldn't find a plant, return source image
        cropped = img

    # transforming image from BGR to RGB and to PIL format
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)
    return pil_img
