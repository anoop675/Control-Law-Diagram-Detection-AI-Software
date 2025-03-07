import cv2
import numpy as np

def extend_line(x1, y1, x2, y2, extension_factor):
    """Extend a line slightly in the direction it's pointing."""
    extend_x = int((x2 - x1) * extension_factor)
    extend_y = int((y2 - y1) * extension_factor)
    return x1 - extend_x, y1 - extend_y, x2 + extend_x, y2 + extend_y

def detect_lines(image_path, theta_values):
  #Keep changing (fine tuning) the threshold , min_line_length, max_line_gap to correctly detect the lines
    """Detect lines using different theta values for horizontal & vertical lines."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    rho = 1  # Distance resolution in pixels
    threshold = 60  # Number of intersections
    min_line_length = 50
    max_line_gap = 31

    detected_lines = []
    
    for theta in theta_values:
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x1, y1, x2, y2 = extend_line(x1, y1, x2, y2, extension_factor=0.005)
                detected_lines.append(((x1, y1), (x2, y2)))

    return detected_lines, edges

# Load and process the image with both horizontal & vertical line detection
image_path = "diagram lines sharpened.jpg"
theta_values = [np.pi/90, np.pi/2]  # Horizontal & Vertical line detection
lines, edges = detect_lines(image_path, theta_values)

img = cv2.imread(image_path)
for line in list(set(lines)):
      (x1, y1), (x2, y2) = line
      cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
      print(line)

output_path = f"detected_lines.jpg"
cv2.imwrite(output_path, img)
'''
# Draw detected lines
i = 1
for line in lines:
      img = cv2.imread(image_path)
      (x1, y1), (x2, y2) = line
      cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
      output_path = f"detected_lines{i}.jpg"
      cv2.imwrite(output_path, img)
      i = i + 1
'''
print(f"Processed image saved at: {output_path}")
