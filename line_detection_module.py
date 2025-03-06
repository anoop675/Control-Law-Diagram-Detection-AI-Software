import cv2
import numpy as np
from google.colab.patches import cv2_imshow
#from google.colab.patches import cv2_imrite
import matplotlib.pyplot as plt

def extend_line(x1, y1, x2, y2, extension_factor):
    """Extend a line slightly in the direction it's pointing."""
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    extend_x = int((x2 - x1) * extension_factor)
    extend_y = int((y2 - y1) * extension_factor)
    return x1 - extend_x, y1 - extend_y, x2 + extend_x, y2 + extend_y

def detect_lines_with_endpoints(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to reduce noise
    #blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive Thresholding for better edge detection
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                               cv2.THRESH_BINARY, 11, 2)

    # Dynamic threshold selection for Canny using Otsu's method
    #high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #low_thresh = 0.5 * high_thresh
    #edges = cv2.Canny(thresh, low_thresh, high_thresh, apertureSize=3)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 28 #28 #200
    min_line_length = 90 #80 #220 #minimum number of pixels making up a line
    max_line_gap = 10

    # Perform Hough Line Transform with adjusted parameters
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

# Display detected edges using matplotlib
#plt.imshow(edges, cmap='gray')
#plt.title("Edges")
#plt.show()

# Draw detected lines on the original image
img = cv2.imread(image_path)
for line in lines:
    (x1, y1), (x2, y2) = line
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Save the output instead of displaying (useful in headless environments)
cv2.imwrite("detected_lines.jpg", img)

