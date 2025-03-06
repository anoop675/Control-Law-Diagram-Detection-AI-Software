import cv2 
import numpy as np
import matplotlib.pyplot as plt

#Increase brightness using gamma correction without overexposing."""
def apply_gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

#Extend a line slightly in the direction it's pointing.
def extend_line(x1, y1, x2, y2, extension_factor):
    extend_x = int((x2 - x1) * extension_factor)
    extend_y = int((y2 - y1) * extension_factor)
    return x1 - extend_x, y1 - extend_y, x2 + extend_x, y2 + extend_y

def detect_lines_with_endpoints(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply gamma correction for brightness enhancement
    img = apply_gamma_correction(img, gamma=1.5)

    # Apply Canny Edge Detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 28  # Adjust threshold for line detection
    min_line_length = 90  # Minimum number of pixels making up a line
    max_line_gap = 10

    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1, y1, x2, y2 = extend_line(x1, y1, x2, y2, extension_factor=0.005)
            detected_lines.append(((x1, y1), (x2, y2)))

    return detected_lines, edges

image_path = "diagram lines sharpened.jpg"
lines, edges = detect_lines_with_endpoints(image_path)

plt.imshow(edges, cmap='gray')
plt.title("Edges after Brightening with Gamma Correction")
plt.show()

img = cv2.imread(image_path)
for line in lines:
    (x1, y1), (x2, y2) = line
    # Draw detected lines on the original image
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("detected_lines.jpg", img)
