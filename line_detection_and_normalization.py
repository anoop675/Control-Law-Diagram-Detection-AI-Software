# TODO: further refinement
'''
  Code for detecting lines using Hough Line Transform and normalizing lines
'''
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Extending a line slightly in the direction it's pointing towards
def extend_line(x1, y1, x2, y2, extension_factor):
    extend_x = int((x2 - x1) * extension_factor)
    extend_y = int((y2 - y1) * extension_factor)
    return x1 - extend_x, y1 - extend_y, x2 + extend_x, y2 + extend_y

# Line detection algorithm using Hough Line Transform
'''
  Image -> Grayscale Conversion -> Gaussian Blur -> Canny Edge Detection -> Morphological Operations -> Hough Line Transform -> Detected lines
'''
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

# Function to check if two line segments are almost similar in the image space
def is_similar(line1, line2, threshold=3):
    """Check if two lines are close enough to be merged"""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # Check if lines are horizontally aligned
    if abs(y1 - y3) <= threshold and abs(y2 - y4) <= threshold:
        return True
    # Check if lines are vertically aligned
    if abs(x1 - x3) <= threshold and abs(x2 - x4) <= threshold:
        return True
    return False

# Function to group similar lines (located close to each very other
def group_and_merge(lines, threshold=3):
    """Group similar lines and merge them into representative lines"""
    used = set()
    merged_lines = []

    for i, line in enumerate(lines):
        if i in used:
            continue

        group = [line]
        used.add(i)

        for j, other_line in enumerate(lines):
            if j in used:
                continue
            if is_similar(line, other_line, threshold):
                group.append(other_line)
                used.add(j)

        # Compute merged line
        x1_vals = [l[0][0] for l in group]
        y1_vals = [l[0][1] for l in group]
        x2_vals = [l[1][0] for l in group]
        y2_vals = [l[1][1] for l in group]

        if abs(np.mean(y1_vals) - np.mean(y2_vals)) < abs(np.mean(x1_vals) - np.mean(x2_vals)):
            # If it's a horizontal line, keep min x1 and max x2
            x1_final = min(x1_vals)
            y1_final = int(np.mean(y1_vals))
            x2_final = max(x2_vals)
            y2_final = int(np.mean(y2_vals))
        else:
            # If it's a vertical line, keep min y1 and max y2
            x1_final = int(np.mean(x1_vals))
            y1_final = min(y1_vals)
            x2_final = int(np.mean(x2_vals))
            y2_final = max(y2_vals)

        merged_lines.append(((x1_final, y1_final), (x2_final, y2_final)))

    return merged_lines

# Function to compute Euclidean distance
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function for clustering lines based on a 6-pixel proximity rule
def form_clusters(lines):
    clusters = []
    visited = set()
    
    for i, line1 in enumerate(lines):
        if i in visited:
            continue
        cluster = [line1]
        visited.add(i)
        
        for j, line2 in enumerate(lines):
            if j in visited:
                continue
            # Compare endpoints of the two lines
            if (distance(line1[0], line2[0]) <= 6 or distance(line1[0], line2[1]) <= 6 or
                distance(line1[1], line2[0]) <= 6 or distance(line1[1], line2[1]) <= 6):
                cluster.append(line2)
                visited.add(j)
        
        clusters.append(cluster)
    
    return clusters

if __name__ == "__main__":
  # Load and process the image
  image_path = "diagram lines.jpg"
  theta_values = [np.pi/90, np.pi]
  lines, edges = detect_lines(image_path, theta_values)
  
  clusters = form_clusters(lines)
  #clustered_lines = {f"Group {i+1}": cluster for i, cluster in clusters}
  #print(clusters)
  normalized_lines = []
  for cluster in clusters:
      normalized_lines.append(group_and_merge(cluster))
  
  '''
  for line in normalized_lines:
      print(line)
  '''

  flat_list = [line for cluster in normalized_lines for line in cluster]
  #lines = group_and_merge(flat_list)
  
  # Draw detected lines
  i = 1
  for ((x1, y1), (x2, y2)) in sorted(flat_list):
      image = cv2.imread(image_path)
      cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color & thickness = 2
      output_path = f"lines{i}.jpg"
      cv2.imwrite(output_path, image)
      print(f"lines{i}.jpg -> ",(x1, y1), (x2, y2))
      i = i + 1
  
  image = cv2.imread(image_path)
  for ((x1, y1), (x2, y2)) in sorted(flat_list):
      cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color & thickness = 2

  # Save or show the output image
  cv2_imshow(image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
