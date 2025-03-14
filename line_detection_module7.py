import cv2
import numpy as np

def extend_line(x1, y1, x2, y2, extension_factor):
    """Extend a line slightly in the direction it's pointing."""
    extend_x = int((x2 - x1) * extension_factor)
    extend_y = int((y2 - y1) * extension_factor)
    return x1 - extend_x, y1 - extend_y, x2 + extend_x, y2 + extend_y

def skeletonize(image):
    """Perform skeletonization on the given image."""
    binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]  # Convert to binary
    skeleton = cv2.ximgproc.thinning(binary)  # Apply skeletonization
    return skeleton

def detect_lines(image_path, theta_values):
    """Detect lines using different theta values for horizontal & vertical lines."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    edges = cv2.Canny(blurred, 40, 170, apertureSize=3)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Hough Transform Parameters
    rho = 1
    threshold = 20
    min_line_length = 16
    max_line_gap = 10

    detected_lines = []
    for theta in theta_values:
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x1, y1, x2, y2 = extend_line(x1, y1, x2, y2, extension_factor=0.005)
                detected_lines.append(((x1, y1), (x2, y2)))

    return detected_lines, edges

# Load and process the image
image_path = "diagram lines.jpg"
theta_values = [np.pi/90, np.pi]
lines, edges = detect_lines(image_path, theta_values)

# Perform skeletonization
skeletonized = skeletonize(edges)

# Convert skeleton to a 3-channel image for overlay
skeleton_colored = cv2.cvtColor(skeletonized, cv2.COLOR_GRAY2BGR)

# Draw detected lines on top of the skeletonized image
for (x1, y1), (x2, y2) in lines:
    cv2.line(skeleton_colored, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines

# Save results
cv2.imwrite("skeletonized_with_lines.jpg", skeleton_colored)

print("Skeletonization with detected lines completed. Saved as skeletonized_with_lines.jpg")
for line in lines:
  print(line, end=",\n")
