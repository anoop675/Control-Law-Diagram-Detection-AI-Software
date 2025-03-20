import math
from collections import defaultdict, deque

# Define a proximity threshold (gap allowed)
EPSILON = 5  

# Given lines: ((start_x, start_y), (end_x, end_y))
lines = [
    ((209, 391), (513, 391)),
    ((211, 257), (640, 257)),
    ((211, 300), (640, 300)),
    ((213, 347), (641, 347)),
    ((215, 129), (594, 129)),
    ((230, 630), (230, 389)),
    ((230, 632), (626, 632)),
    ((341, 192), (369, 192)),
    ((343, 475), (465, 475)),
    ((367, 154), (592, 154)),
    ((367, 193), (367, 154)),
    ((447, 551), (661, 551)),
    ((557, 393), (642, 393)),
    ((559, 476), (619, 476)),
    ((599, 597), (615, 597)),
    ((613, 585), (662, 585)),
    ((614, 599), (614, 583)),
    ((616, 460), (618, 437)),
    ((618, 474), (618, 440)),
    ((622, 610), (791, 610)),
    ((624, 630), (624, 612)),
    ((670, 141), (965, 141)),
    ((734, 572), (791, 572)),
    ((756, 295), (1103, 295)),
    ((762, 404), (1103, 404)),
    ((774, 224), (817, 224)),
    ((775, 225), (775, 299)),
    ((832, 77), (898, 77)),
    ((844, 590), (886, 590)),
    ((892, 223), (928, 223)),
    ((897, 76), (897, 116)),
    ((897, 115), (966, 115)),
    ((926, 224), (926, 169)),
    ((927, 167), (967, 167)),
    ((929, 591), (1104, 591)),
    ((1022, 142), (1101, 142))
]

# Function to compute Euclidean distance
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Store updated lines
updated_lines = lines.copy()

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

# Print updated connections
print("\nUpdated Lines:")
for i, line in enumerate(updated_lines):
    print(f"Line {i+1}: {line}")
