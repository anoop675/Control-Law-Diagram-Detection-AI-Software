import cv2
import math
import numpy as np
from google.colab.patches import cv2_imshow
from typing import Final

# Extending a line slightly in the direction it's pointing towards
def extend_line(x1, y1, x2, y2):
    EXTENSION_FACTOR: Final = 0.00
    extend_x = int((x2 - x1) * EXTENSION_FACTOR)
    extend_y = int((y2 - y1) * EXTENSION_FACTOR)
    return x1 - extend_x, y1 - extend_y, x2 + extend_x, y2 + extend_y

# Line detection algorithm using Hough Line Transform
'''
  Image -> Grayscale Conversion -> Gaussian Blur / Bilateral Filtering -> Canny Edge Detection -> Morphological Operations -> Hough Line Transform -> Detected lines
'''
def detect_lines(image_path):
    theta_values = [np.pi/90, np.pi] #for detecting horizontal and vertical lines.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)[1]
    #blurred = cv2.GaussianBlur(image, (5,5), 0)
    #blurred = cv2.blur(img,(5,5))
    
    DIAMETER_OF_PIXEL_NEIGHBOURHOOD: Final = 6
    SIGMA_COLOR: Final = 50
    SIGMA_SPACE: Final = 50
    blurred = cv2.bilateralFilter(img, DIAMETER_OF_PIXEL_NEIGHBOURHOOD, SIGMA_COLOR, SIGMA_SPACE, cv2.BORDER_DEFAULT)

    LOWER_HYSTERISIS_THRESHOLD: Final = 40
    HIGHER_HYSTERISIS_THRESHOLD: Final = 170
    APERTURE_SIZE: Final = 3
    edges = cv2.Canny(blurred, LOWER_HYSTERISIS_THRESHOLD, HIGHER_HYSTERISIS_THRESHOLD, APERTURE_SIZE)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Hough Transform Parameters
    RHO: Final = 1
    THRESHOLD: Final = 20
    MIN_LINE_LENGTH: Final = 16
    MAX_LINE_GAP: Final = 10

    detected_lines = []
    for theta in theta_values:
        lines = cv2.HoughLinesP(edges, RHO, theta, THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x1, y1, x2, y2 = extend_line(x1, y1, x2, y2)
                detected_lines.append(((x1, y1), (x2, y2)))

    return detected_lines, edges

# Function to check if two line segments are almost similar in the image space, and be merged based on proximity, overlap, and parallelism.
def can_merge(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    THRESHOLD: Final = 5

    # Check if both lines are approximately horizontal
    if abs(y1 - y2) <= THRESHOLD and abs(y3 - y4) <= THRESHOLD:
        if abs(y1 - y3) <= THRESHOLD:  # Close in y-coordinates
            if max(x1, x3) <= min(x2, x4) or abs(x1 - x3) <= THRESHOLD or abs(x2 - x4) <= THRESHOLD:
                return True

    # Check if both lines are approximately vertical
    if abs(x1 - x2) <= THRESHOLD and abs(x3 - x4) <= THRESHOLD:
        if abs(x1 - x3) <= THRESHOLD:  # Close in x-coordinates
            if max(y1, y3) <= min(y2, y4) or abs(y1 - y3) <= THRESHOLD or abs(y2 - y4) <= THRESHOLD:
                return True

    return False

# Function which merges two overlapping or close lines into a single extended line.
def merge_lines(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    THRESHOLD: Final = 5

    # For horizontal lines, merge by extending the x-range
    if abs(y1 - y2) <= THRESHOLD and abs(y3 - y4) <= THRESHOLD:
        y_avg = int((y1 + y3) / 2)  # Take the average y-value
        return (min(x1, x3), y_avg), (max(x2, x4), y_avg)

    # For vertical lines, merge by extending the y-range
    if abs(x1 - x2) <= THRESHOLD and abs(x3 - x4) <= THRESHOLD:
        x_avg = int((x1 + x3) / 2)  # Take the average x-value
        return (x_avg, min(y1, y3)), (x_avg, max(y2, y4))

    return line1  # If no merging is possible, return the original line

#Function that identifies, groups and merges lines based on proximity, overlap, and parallelism.
def group_and_merge(lines):
    merged_lines = []
    used = set()
    THRESHOLD: Final = 5

    for i, line1 in enumerate(lines):
        if i in used:
            continue

        for j, line2 in enumerate(lines):
            if i != j and j not in used and can_merge(line1, line2):
                line1 = merge_lines(line1, line2)  # Merge the lines
                used.add(j)  # Mark the second line as used

        merged_lines.append(line1)  # Store the final merged line

    return merged_lines

def merge_lines2(lines):
    merged_lines = []
    THRESHOLD: Final = 5

    for line in lines:
        (x1, y1), (x2, y2) = line  # Unpacking directly from tuple-of-tuples
        merged = False

        for j in range(len(merged_lines)):
            (mx1, my1), (mx2, my2) = merged_lines[j]  # Unpacking existing merged lines

            # Case 1: Merging Horizontal Lines
            if abs(y1 - my1) <= THRESHOLD and abs(y2 - my2) <= THRESHOLD:
                mean_y = (y1 + my1) // 2  # Compute average y
                if mx1 <= x1 <= mx2 or mx1 <= x2 <= mx2 or x1 <= mx1 <= x2:  # Overlapping X
                    new_x1 = min(mx1, x1, x2, mx2)
                    new_x2 = max(mx1, x1, x2, mx2)
                    merged_lines[j] = ((new_x1, mean_y), (new_x2, mean_y))
                    merged = True
                    break

            # Case 2: Merging Vertical Lines
            if abs(x1 - mx1) <= THRESHOLD and abs(x2 - mx2) <= THRESHOLD:
                mean_x = (x1 + mx1) // 2  # Compute average x
                if my1 <= y1 <= my2 or my1 <= y2 <= my2 or y1 <= my1 <= y2:  # Overlapping Y
                    new_y1 = min(my1, y1, y2, my2)
                    new_y2 = max(my1, y1, y2, my2)
                    merged_lines[j] = ((mean_x, new_y1), (mean_x, new_y2))
                    merged = True
                    break
        
        if not merged:
            merged_lines.append(((x1, y1), (x2, y2)))  # Append as a new independent line

    return merged_lines  # Keeping original tuple-of-tuple format

# Function to compute Euclidean distance
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function for clustering lines based on a 6-pixel proximity rule
def form_clusters(lines):
    clusters = []
    visited = set()
    PROXIMITY: Final = 6 #6 pixels
    
    for i, line1 in enumerate(lines):
        if i in visited:
            continue
        cluster = [line1]
        visited.add(i)
        
        for j, line2 in enumerate(lines):
            if j in visited:
                continue
            # Compare endpoints of the two lines
            if (distance(line1[0], line2[0]) <= PROXIMITY or distance(line1[0], line2[1]) <= PROXIMITY or
                distance(line1[1], line2[0]) <= PROXIMITY or distance(line1[1], line2[1]) <= PROXIMITY):
                cluster.append(line2)
                visited.add(j)
        
        clusters.append(cluster)
    
    return clusters
'''
# Function to check if two points are close enough to be considered connected
def is_close(p1, p2, threshold):
    return math.dist(p1, p2) <= threshold

# Function to check if two lines are connected
def are_lines_connected(line1, line2, threshold):
    (x1, y1), (x2, y2) = line1
    (a1, b1), (a2, b2) = line2

    # Check if any endpoint of line1 is close to any endpoint of line2
    is_connected = is_close((x1, y1), (a1, b1), threshold) or is_close((x1, y1), (a2, b2), threshold) \
                    or is_close((x2, y2), (a1, b1), threshold) or is_close((x2, y2), (a2, b2), threshold)
    return is_connected
'''

if __name__ == "__main__":
  # Load and process the image
  image_path = "diagram lines.jpg"
  lines, edges = detect_lines(image_path)
  
  normalized_lines = []
  clusters = form_clusters(lines) #used to group unecessary & noisy lines and normalize it into a single line based on proximity
  #clustered_lines = {f"Group {i+1}": cluster for i, cluster in clusters}
  #print(clusters)
  for cluster in clusters:
      normalized_lines.append(group_and_merge(cluster)) #first-pass of merging lines

  flat_list = [line for cluster in normalized_lines for line in cluster]
  lines2 = merge_lines2(flat_list) #second-pass of merging lines

  '''
  # Draw detected lines
  i = 1
  for ((x1, y1), (x2, y2)) in sorted(lines):
      image = cv2.imread(image_path)
      cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color & thickness = 2
      output_path = f"lines{i}.jpg"
      cv2.imwrite(output_path, image)
      print(f"lines{i} -> ",(x1, y1), (x2, y2))
      i = i + 1
  '''
  image = cv2.imread(image_path)
  for ((x1, y1), (x2, y2)) in sorted(lines2):
      cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color & thickness = 2
      
      # Draw points on both endpoints
      #cv2.circle(image, (x1, y1), 5, (0, 255, 0), -1)
      #cv2.circle(image, (x2, y2), 5, (0, 255, 0), -1)  
      
  output_path = "lines.jpg"
  cv2.imwrite(output_path, image)
  cv2_imshow(image)
  # Save or show the output image
  cv2.waitKey(0)
  cv2.destroyAllWindows()
