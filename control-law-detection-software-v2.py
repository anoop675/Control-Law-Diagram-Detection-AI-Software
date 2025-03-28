import cv2
import math
import numpy as np
from google.colab.patches import cv2_imshow
from collections import defaultdict, deque
from typing import Final, List, Tuple

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

    DIAMETER_OF_PIXEL_NEIGHBOURHOOD: Final = 5
    SIGMA_COLOR: Final = 28
    SIGMA_SPACE: Final = 28
    # 8 80 80   9 85 85  9 80 80
    blurred = cv2.bilateralFilter(img, DIAMETER_OF_PIXEL_NEIGHBOURHOOD, SIGMA_COLOR, SIGMA_SPACE, cv2.BORDER_DEFAULT)

    LOWER_HYSTERISIS_THRESHOLD: Final = 50
    HIGHER_HYSTERISIS_THRESHOLD: Final = 200
    APERTURE_SIZE: Final = 3
    # 40 170 3
    edges = cv2.Canny(blurred, LOWER_HYSTERISIS_THRESHOLD, HIGHER_HYSTERISIS_THRESHOLD, APERTURE_SIZE)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Hough Transform Parameters
    RHO: Final = 1
    THRESHOLD: Final = 12
    MIN_LINE_LENGTH: Final = 10
    MAX_LINE_GAP: Final = 8

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
        (x1, y1), (x2, y2) = line
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Convert to Python integers
        merged = False

        for j in range(len(merged_lines)):
            (mx1, my1), (mx2, my2) = merged_lines[j]
            mx1, my1, mx2, my2 = map(int, (mx1, my1, mx2, my2))  # Convert existing lines to Python integers

            # Case 1: Merging Horizontal Lines
            if abs(y1 - my1) <= THRESHOLD and abs(y2 - my2) <= THRESHOLD:
                mean_y = int((y1 + my1) // 2)  # Compute average y
                if mx1 <= x1 <= mx2 or mx1 <= x2 <= mx2 or x1 <= mx1 <= x2:  # Overlapping X
                    new_x1 = int(min(mx1, x1, x2, mx2))
                    new_x2 = int(max(mx1, x1, x2, mx2))
                    merged_lines[j] = ((new_x1, mean_y), (new_x2, mean_y))
                    merged = True
                    break

            # Case 2: Merging Vertical Lines
            if abs(x1 - mx1) <= THRESHOLD and abs(x2 - mx2) <= THRESHOLD:
                mean_x = int((x1 + mx1) // 2)  # Compute average x
                if my1 <= y1 <= my2 or my1 <= y2 <= my2 or y1 <= my1 <= y2:  # Overlapping Y
                    new_y1 = int(min(my1, y1, y2, my2))
                    new_y2 = int(max(my1, y1, y2, my2))
                    merged_lines[j] = ((mean_x, new_y1), (mean_x, new_y2))
                    merged = True
                    break

        if not merged:
            merged_lines.append(((x1, y1), (x2, y2)))

    return merged_lines

# Function for clustering lines based on a 6-pixel proximity rule
def form_clusters(lines):
    # Nested Function to compute Euclidean distance
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    clusters = []
    visited = set()
    PROXIMITY: Final = 5 #6 pixels

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

def update_lines_to_connect(lines):

  def distance(p1, p2):
      return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

  updated_lines = lines.copy()
  EPSILON: Final = 5
  # Mapping from line index to adjusted coordinates
  adjusted_points = {}

  # Iterate over all pairs of lines
  for i, ((x1a, y1a), (x2a, y2a)) in enumerate(lines):
      for j, ((x1b, y1b), (x2b, y2b)) in enumerate(lines):
          if i != j:  # Avoid self-comparison

              # Check if two endpoints are within the EPSILON threshold
              if distance((x1a, y1a), (x1b, y1b)) <= EPSILON:
                  adjusted_points[(i, "start")] = (x1b, y1b)
                  adjusted_points[(j, "start")] = (x1b, y1b)

              elif distance((x1a, y1a), (x2b, y2b)) <= EPSILON:
                  adjusted_points[(i, "start")] = (x2b, y2b)
                  adjusted_points[(j, "end")] = (x2b, y2b)

              elif distance((x2a, y2a), (x1b, y1b)) <= EPSILON:
                  adjusted_points[(i, "end")] = (x1b, y1b)
                  adjusted_points[(j, "start")] = (x1b, y1b)

              elif distance((x2a, y2a), (x2b, y2b)) <= EPSILON:
                  adjusted_points[(i, "end")] = (x2b, y2b)
                  adjusted_points[(j, "end")] = (x2b, y2b)

  # Apply adjustments
  for (line_idx, position), new_point in adjusted_points.items():
      start, end = updated_lines[line_idx]
      if position == "start":
          updated_lines[line_idx] = (new_point, end)
      else:
          updated_lines[line_idx] = (start, new_point)

  return updated_lines

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def normalize_points(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Normalize points by reducing small variations and sorting consistently."""
    VARIATION_THRESHOLD = 2  # Max allowable x/y variation (2 pixels)
    points = sorted(points, key=lambda p: (p[0], p[1]))  # Sort by x, then y
    
    normalized = []
    for x, y in points:
        if normalized:
            prev_x, prev_y = normalized[-1]
            if abs(x - prev_x) <= VARIATION_THRESHOLD:
                x = prev_x
            if abs(y - prev_y) <= VARIATION_THRESHOLD:
                y = prev_y
        normalized.append((x, y))
    return normalized

def standardize_line_segments(segments: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Ensure horizontal lines are left-to-right and vertical lines are top-to-bottom."""
    standardized = []
    for start, end in segments:
        x1, y1 = start
        x2, y2 = end
        if x1 == x2:  # Vertical line
            if y1 > y2:
                start, end = end, start
        elif y1 == y2:  # Horizontal line
            if x1 > x2:
                start, end = end, start
        standardized.append((start, end))
    return standardized

def build_graph(line_segments):
    """Creates an adjacency list representation of the graph."""
    graph = defaultdict(set)
    for start, end in line_segments:
        graph[start].add(end)
        graph[end].add(start)
    return {key: list(value) for key, value in graph.items()}

def find_connected_paths(line_segments, epsilon=5):
    """Finds connected paths from line segments with consistent ordering."""
    graph = defaultdict(set)
    corrected_points = {}
    
    def get_corrected_point(point):
        """Find the closest existing point within threshold, or use itself."""
        for existing in corrected_points:
            if distance(point, existing) <= epsilon:
                return corrected_points[existing]
        corrected_points[point] = point
        return point
    
    # Build adjacency list with corrected points
    for start, end in line_segments:
        start = get_corrected_point(start)
        end = get_corrected_point(end)
        graph[start].add(end)
        graph[end].add(start)
    
    # Find connected components using DFS with directionality
    visited = set()
    paths = []
    
    def dfs(node, path):
        if node in visited:
            return
        visited.add(node)
        path.append(node)
        neighbors = sorted(graph[node], key=lambda p: (p[0], p[1]))  # Ensure consistent traversal
        for neighbor in neighbors:
            dfs(neighbor, path)
    
    for node in sorted(graph.keys(), key=lambda p: (p[0], p[1])):
        if node not in visited:
            path = []
            dfs(node, path)
            paths.append(path)
    
    return paths

def is_inside_bbox(point, bbox, tolerance=3):
    """Checks if a point (x, y) is inside or very close to a bounding box."""
    x, y = point
    x_min, y_min, x_max, y_max = bbox

    return (x_min - tolerance) <= x <= (x_max + tolerance) and (y_min - tolerance) <= y <= (y_max + tolerance)

def find_existing_source(coinciding_point, objects, connections, path_list, tolerance=5):
    """
    Finds an existing source for a coinciding point by first checking strict alignment,
    then allowing a tolerance-based search.
    """
    source = None

    # Step 1: Check for exact path alignment (strict check)
    for existing_path in path_list:
        start, end = existing_path[0], existing_path[-1]

        if ((coinciding_point[1] == start[1] == end[1]) and (start[0] <= coinciding_point[0] <= end[0])) or \
           ((coinciding_point[0] == start[0] == end[0]) and (start[1] <= coinciding_point[1] <= end[1])):
            
            # Look for an object bounding box containing the path start
            for obj, data in objects.items():
                if is_inside_bbox(start, data["bbox"]):
                    return obj  # Return source immediately if found
    
    # Step 2: If strict check fails, perform tolerance-based check
    for existing_path in path_list:
        start, end = existing_path[0], existing_path[-1]

        if (abs(coinciding_point[1] - start[1]) <= tolerance and abs(coinciding_point[1] - end[1]) <= tolerance and 
            min(start[0], end[0]) - tolerance <= coinciding_point[0] <= max(start[0], end[0]) + tolerance) or \
           (abs(coinciding_point[0] - start[0]) <= tolerance and abs(coinciding_point[0] - end[0]) <= tolerance and 
            min(start[1], end[1]) - tolerance <= coinciding_point[1] <= max(start[1], end[1]) + tolerance):
            
            # Look for an object bounding box containing the path start
            for obj, data in objects.items():
                if is_inside_bbox(start, data["bbox"]):
                    return obj  # Return source if found within tolerance

    return None  # No match found

def find_fork_source(fork_point, path_list):
    """Identify the true source of a fork point based on previous paths."""
    for path in path_list:
        if fork_point in path and path.index(fork_point) > 0:
            return path[0]  # The true source of this fork
    return None

def detect_object(point, objects, connections, path_list):
    """Determine the source for a given point."""
    for obj, data in objects.items():
        if is_inside_bbox(point, data["bbox"]):
            return obj  # Direct source found

    # If no direct source, check if it's a fork
    fork_source = find_fork_source(point, path_list)
    if fork_source:
        return fork_source  # Prioritize forks

    # Finally, check for previous paths
    return find_existing_source(point, objects, connections, path_list)

def build_dag(objects, connections):
    """Build a Directed Acyclic Graph (DAG) using objects and path connections."""
    graph = defaultdict(list)
    path_list = []  # Store all paths for reference
    
    for path in connections:
        # Normalize the path first
        #path = normalize_path(path, objects)
        
        source, destination = None, None

        # Detect the source of the path
        for point in path:
            potential_source = detect_object(point, objects, connections, path_list)
            if potential_source:
                source = potential_source
                break

        # Detect the destination of the path
        '''
        for point in reversed(path):
            potential_destination = detect_object(point, objects, connections, path_list)
            if potential_destination and potential_destination != source:
                destination = potential_destination
                break
        '''
        for point in reversed(path): #reversed(path) because the we are considering the last point of the path
            potential_destination = None
            
            #Checking if ending point point of the path is connected to a source object
            for obj, data in objects.items():
                if is_inside_bbox(point, data["bbox"]):
                    potential_destination = obj
                    
                    if destination is None or potential_destination != source:  
                        destination = potential_destination  
                        print(f"Potential Destination: {potential_destination} for point {point}")
                        
            if destination:
                break

        # Add edge to DAG if valid
        if source and destination and source != destination:
            if destination not in graph[source]:  # Avoid duplicate edges
                graph[source].append(destination)

        # Store path for future fork detection
        path_list.append(path)

    # Ensure all detected output nodes exist in graph
    all_destinations = {dest for dest_list in graph.values() for dest in dest_list}
    for dest in all_destinations:
        if dest not in graph:
            graph[dest] = []

    return dict(graph)

# Topological Sort to get the order of function calls
def topological_sort(graph):
    topo_order = []
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in graph if in_degree[node] == 0])

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    #print("Topological sorted order: "+str(topo_order))
    return topo_order

# Generate C code from topological order
def generate_c_code(graph):
    topo_order = topological_sort(graph)  # Get nodes in dependency order
    c_code = [
         "#include <stdio.h>", "#include <stdlib.h>", "#include <float.h>", "#include <stdbool.h>", 
         "#include <string.h>", "#include <math.h>\n"
     ]
    c_code.append("// TODO: function definitions here")
     
    c_code.append("\nint main(void) {")
     
    temp_var_count = 1
    temp_vars = {}  # To map logical functions to temporary variables
    output_nodes = []
    variable_declarations = []  # Store variable declarations separately
    assignments = []  # Store assignments separately
 
    for node in topo_order:
        if node in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}:  # Input nodes
            variable_declarations.append(f"    double {node};  // Input variable")
            continue
 
        inputs = [var for var in graph if node in graph[var]]  # Get input nodes
        input_vars = [temp_vars.get(i, i) for i in inputs]  # Replace with temp vars if needed
 
        if node.startswith("Y"):  # Output nodes
            variable_declarations.append(f"    double {node};  // Output variable")
            if len(input_vars) == 1:
                assignments.append(f"    {node} = {input_vars[0]};")
            else:
                assignments.append(f"    {node};  // Undefined behavior detected")
            output_nodes.append(node)
        else:  # Intermediate computations (e.g., OR, NOT)
            temp_var = f"temp{temp_var_count}"
            temp_vars[node] = temp_var
            temp_var_count += 1
            variable_declarations.append(f"    double {temp_var};  // Temporary variable")
            assignments.append(f"    {temp_var} = {node}({', '.join(input_vars)});")
 
    # Insert variable declarations first
    c_code.extend(variable_declarations)
    c_code.append("")  # Add a blank line for readability
     
    # Insert assignments later
    c_code.extend(assignments)
     
    # Print statements for output
    for output in output_nodes:
        c_code.append(f'    printf("{output} = %f\\n", {output});')
 
    c_code.append("\n    return 0;")
    c_code.append("}")
 
    return "\n".join(c_code)

if __name__ == "__main__":
    try:
        objects = {
          "A": {"bbox": (131, 111, 215, 150)},
          "B": {"bbox": (252, 181, 341, 216)},
          "C": {"bbox": (140, 240, 211, 270)},
          "D": {"bbox": (140, 281, 211, 323)},
          "E": {"bbox": (140, 330, 213, 364)},
          "F": {"bbox": (140, 370, 209, 420)},
          "G": {"bbox": (275, 452, 343, 500)},
          "H": {"bbox": (341, 520, 447, 583)},
          "I": {"bbox": (564, 567, 599, 631)},
          "J": {"bbox": (796, 54, 832, 120)},
          "LTH": {"bbox": (592, 93, 670, 190)},
          "NOT1": {"bbox": (513, 360, 557, 425)},
          "FUN316": {"bbox": (465, 440, 559, 520)},
          "GRT": {"bbox": (661, 522, 734, 600)},
          "DEC": {"bbox": (791, 550, 844, 640)},
          "NOT2": {"bbox": (886, 560, 929, 630)},
          "FUN317": {"bbox": (817, 198, 892, 245)},
          "BBANG": {"bbox": (965, 80, 1022, 207)},
          "INTEGRATOR": {"bbox": (640, 220, 762, 470)},
          "Y1": {"bbox": (1101, 102, 1141, 172)},
          "Y2": {"bbox": (1103, 255, 1143, 345)},
          "Y3": {"bbox": (1103, 364, 1143, 444)},
          "Y4": {"bbox": (1104, 561, 1144, 651)}
        }
        '''
        line_segments = [
            ((557, 393), (642, 393)),
            ((670, 141), (965, 141)),
            ((341, 192), (369, 192)),
            ((1022, 142), (1101, 142)),
            ((927, 167), (967, 167)),
            ((762, 404), (1103, 404)),
            ((844, 590), (886, 590)),
            ((774, 224), (817, 224)),
            ((367, 154), (592, 154)),
            ((622, 610), (791, 610)),
            ((756, 295), (1103, 295)),
            ((211, 300), (640, 300)),
            ((892, 223), (928, 223)),
            ((614, 599), (614, 583)),
            ((209, 391), (513, 391)),
            ((616, 460), (618, 437)),
            ((599, 597), (615, 597)),
            ((343, 475), (465, 475)),
            ((613, 585), (662, 585)),
            ((230, 630), (230, 389)),
            ((215, 129), (594, 129)),
            ((897, 76), (897, 116)),
            ((213, 347), (641, 347)),
            ((230, 632), (626, 632)),
            ((929, 591), (1104, 591)),
            ((618, 474), (618, 440)),
            ((832, 77), (898, 77)),
            ((211, 257), (640, 257)),
            ((775, 225), (775, 295)),
            ((624, 630), (624, 612)),
            ((897, 115), (966, 115)),
            ((447, 551), (661, 551)),
            ((734, 572), (791, 572)),
            ((926, 224), (926, 169)),
            ((367, 193), (367, 154)),
            ((559, 476), (619, 476))
        ]'''
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
        #lines3 = connect_lines(lines2)
        lines2 = merge_lines2(lines2)
        #lines3 = merge_lines2(lines2)
        updated_lines = update_lines_to_connect(lines2)

        standardized_segments = standardize_line_segments(updated_lines)
        connected_paths = find_connected_paths(standardized_segments)

        dag = build_dag(objects, connected_paths)
        for item in dag.items():
          print(item) 
        
        #c_code = generate_c_code(dag, function_definitions)
        c_code = generate_c_code(dag)
        #print(c_code)
        with open("Control Law Diagram.c", "w") as file: #TODO: include module identifier for the control law diagram whihc will be shared by DRDO
            file.write(c_code)

    except KeyError as e1:
        print(f"ERROR: {e1}")
