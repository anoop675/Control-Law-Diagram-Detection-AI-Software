from collections import defaultdict

def merge_paths(connections):
    graph = defaultdict(list)

    # Build graph from individual line segments
    for start, end in connections:
        graph[start].append((end, [start, end]))

    merged_paths = []
    visited = set()

    # Depth First Search to merge paths
    def dfs(node, current_path):
        if node in visited:
            return
        visited.add(node)

        # Add current node to the path
        if current_path and current_path[-1] != node:
            current_path.append(node)
        elif not current_path:
            current_path.append(node)

        # Traverse connected segments
        if node in graph:
            for neighbor, segment in graph[node]:
                # Avoid duplicates
                if segment[1:] not in current_path:
                    dfs(neighbor, current_path + segment[1:])
        else:
            # End of the path
            merged_paths.append(current_path)

    # Traverse graph and merge paths
    for start, edges in graph.items():
        if start not in visited:
            for neighbor, segment in edges:
                dfs(neighbor, segment[:])

    # Remove duplicate points in the merged path
    final_paths = []
    for path in merged_paths:
        clean_path = [path[0]]
        for i in range(1, len(path)):
            if path[i] != path[i-1]:
                clean_path.append(path[i])
        final_paths.append(clean_path)

    return final_paths


connections = [
    ((200, 75), (305, 150)),
    ((220, 225), (260, 225)),
    ((260, 225), (260, 175)),
    ((260, 175), (305, 175)),
    ((400, 150), (500, 150)),  
    ((200, 425), (300, 425)), 
    ((350, 425), (500, 425)), 
    ((350, 425), (485, 270)), 
    ((485, 270), (670, 270)),
    ((280, 425), (280, 575)),
    ((280, 575), (600, 575)),
    ((425, 425), (425, 150)),
    ((425, 150), (650, 312)),
    ((530, 575), (530, 600)),
    ((530, 600), (730, 600)),
    ((730, 600), (730, 450)),
    ((730, 450), (650, 450)),
    ((650, 450), (650, 300)),
    ((750, 300), (800, 300)) 
]

# Run the merge_paths function
merged_paths = merge_paths(connections)

# Output the correct logical flow
for path in merged_paths:
    print(path)
