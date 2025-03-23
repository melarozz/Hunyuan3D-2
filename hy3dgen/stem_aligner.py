import cv2
import numpy as np
from PIL import Image

class StemAligner:
    def __init__(self):
        self.final_width = None
        self.final_height = None
        self.M = None

    def detect_stem_x(self, image_np):
        """Detects the stem horizontally (x coordinate) from the largest contour.
           Assumes the stem is near the bottom of the plant.
           Expects a BGR image (numpy array)."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            max_y = np.max(main_contour[:, :, 1])
            threshold = 5
            stem_points = main_contour[main_contour[:, :, 1] >= (max_y - threshold)]
            # Adjust for different possible array shapes
            if stem_points.ndim == 3:
                stem_x = int(np.mean(stem_points[:, 0, 0]))
            else:
                stem_x = int(np.mean(stem_points[:, 0]))
            return stem_x
        # Fallback: use image center if no contour is found
        return image_np.shape[1] // 2

    def preprocess_images(self, images):
        """Extracts image data and computes alignment parameters."""
        image_data = []
        M_candidates = []

        for key in images:
            pil_img = images[key].convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            h, w = cv_img.shape[:2]
            stem_x = self.detect_stem_x(cv_img)

            image_data.append({'key': key, 'img': cv_img, 'w': w, 'h': h, 'stem_x': stem_x})
            M_candidates.extend([stem_x, w - stem_x])

        self.M = max(M_candidates)
        self.final_width = 2 * self.M
        self.final_height = max(d['h'] for d in image_data)

        print("Final canvas size will be:", self.final_width, "x", self.final_height)
        return image_data

    def align_images(self, images):
        """Aligns the images based on the detected stem position."""
        image_data = self.preprocess_images(images)

        for d in image_data:
            img, stem_x = d['img'], d['stem_x']
            dx = self.M - stem_x  # shift needed horizontally

            # Create a white canvas
            canvas = np.ones((self.final_height, self.final_width, 3), dtype=np.uint8) * 255

            # Compute positioning
            canvas_x = max(dx, 0)
            img_x = 0 if dx >= 0 else -dx
            paste_width = min(d['w'] - img_x, self.final_width - canvas_x)
            paste_height = min(d['h'], self.final_height)

            # Place the image into the canvas
            canvas[0:paste_height, canvas_x:canvas_x + paste_width] = \
                img[0:paste_height, img_x:img_x + paste_width]

            # Convert back to PIL Image and update dictionary
            aligned_pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            images[d['key']] = aligned_pil_img

            print(f"Processed {d['key']}: Stem at x={stem_x} moved to x={self.M} (shift of {dx} pixels)")

        return images
