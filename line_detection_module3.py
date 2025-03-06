import cv2 
import numpy as np
import matplotlib.pyplot as plt

def extend_line(x1, y1, x2, y2, extension_factor):
    """Extend a line slightly in the direction it's pointing."""
    extend_x = int((x2 - x1) * extension_factor)
    extend_y = int((y2 - y1) * extension_factor)
    return x1 - extend_x, y1 - extend_y, x2 + extend_x, y2 + extend_y

def detect_lines_with_endpoints(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian Blur to smooth edges
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Apply Morphological Closing to connect gaps
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    rho = 1  # Distance resolution in pixels
    theta = np.pi/180  # Angular resolution in radians
    threshold = 28  # Number of intersections in Hough space
    min_line_length = 90  # Minimum line length
    max_line_gap = 20  # Increased gap to link broken segments
    
    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1, y1, x2, y2 = extend_line(x1, y1, x2, y2, extension_factor=0.005)
            detected_lines.append(((x1, y1), (x2, y2)))
    
    return detected_lines, edges

# Load and process the image
image_path = "diagram lines sharpened.jpg"
lines, edges = detect_lines_with_endpoints(image_path)

# Draw detected lines on the original image
img = cv2.imread(image_path)
for line in lines:
    (x1, y1), (x2, y2) = line
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Save the output image
cv2.imwrite("detected_lines.jpg", img)
