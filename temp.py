from collections import defaultdict

# Function to find continuous paths without breaks
def find_paths(line_segments):
    graph = defaultdict(list)

    # Build a graph where start -> end lines are connected
    for start, end in line_segments:
        graph[start].append(end)

    # Trace all paths from a starting point
    def trace_path(start, visited):
        path = [start]
        current = start
        
        # Traverse the path until break
        while current in graph:
            next_point = graph[current][0]
            if next_point in visited:
                break  # Prevent cyclic paths
            path.append(next_point)
            visited.add(next_point)
            current = next_point

        return path

    # Traverse the graph to find all paths
    paths = []
    visited = set()
    for start in graph:
        if start not in visited:
            path = trace_path(start, visited)
            if len(path) > 1:
                paths.append(path)
    return paths


# Test Data
line_segments = [
    ((650, 450), (650, 300)),
    ((750, 300), (800, 300)),
    ((400, 150), (500, 150)),
    ((200, 425), (300, 425)),
    ((350, 425), (485, 270)),
    ((485, 270), (670, 270)),
    ((280, 425), (280, 575)),
    ((280, 575), (600, 575)),
    ((260, 175), (305, 175)),
    ((220, 225), (260, 225)),
    ((530, 575), (530, 600)),
    ((530, 600), (730, 600)),
    ((730, 600), (730, 450)),
    ((425, 425), (425, 150)),
    ((425, 150), (650, 312)),
    ((200, 150), (305, 150)),
    ((260, 225), (260, 175)),
    ((730, 450), (650, 450))
]

# Generate the paths
paths = find_paths(line_segments)

# Output the result
for path in paths:
    print(path)
