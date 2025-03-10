from collections import defaultdict, deque

def create_logical_paths(lines):
    graph = defaultdict(list)

    # Build graph from line segments
    for path in lines:
        start = path[0]
        end = path[-1]
        graph[start].append((end, path))

    merged_paths = []
    visited = set()

    # Depth First Search to merge paths
    def dfs(node, current_path):
        if node in visited:
            return
        visited.add(node)
        current_path.append(node)

        if node in graph:
            for neighbor, subpath in graph[node]:
                # Avoid duplicates
                if subpath[1:] not in current_path:
                    dfs(neighbor, current_path + subpath[1:])
        else:
            merged_paths.append(current_path)

    # Traverse graph and merge paths
    for start, edges in graph.items():
        if start not in visited:
            for neighbor, path in edges:
                dfs(neighbor, path[:])

    # Remove duplicate points and overlapping segments
    final_paths = []
    for path in merged_paths:
        clean_path = [path[0]]
        for i in range(1, len(path)):
            if path[i] != path[i-1]:
                clean_path.append(path[i])
        final_paths.append(clean_path)

    return final_paths

def is_inside_bbox(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox

    return x_min <= x <= x_max and y_min <= y <= y_max

def get_existing_source(coinciding_point, objects, connections, path_list):
    source = None
    #print(f"Checking for coinciding point: {coinciding_point}")
            
    for existing_path in path_list:
        start, end = existing_path[0], existing_path[-1]
            
        # Check if the coinciding point aligns with an existing path
        if ((coinciding_point[1] == start[1] == end[1]) and (coinciding_point[0] != start[0] != end[0])) or ((coinciding_point[0] == start[0] == end[0]) and (coinciding_point[1] != start[1] != end[1])):
            
            #print(f"Coinciding point {coinciding_point} aligns with an existing path: {existing_path}")
            # Find the source of this existing path
            for obj, data in objects.items():
                if is_inside_bbox(start, data["bbox"]):
                    source = obj  # Assign valid source
                    #print(f"Source found through existing path alignment: {source}")
                    break
            if source:
                break
         
    return source

# Build DAG using objects and connections
def build_dag(objects, connections):
    graph = defaultdict(list)
    path_list = []  # Store all paths for reference

    for path in connections:
        source, destination = None, None
        
        #print(f"Path: {path}")
        # Identify source node (first point inside a bbox)
        for point in path:
            potential_source = None  # Reset for each point
        
            # Check if the point is inside any bounding box
            for obj, data in objects.items():
                if is_inside_bbox(point, data["bbox"]):
                    potential_source = obj
                    break  # Stop checking further if a source is found
        
            # If no direct source is found, check for existing paths
            if potential_source is None:
                potential_source = get_existing_source(point, objects, connections, path_list)
        
            #print(f"Potential Source: {potential_source} for point {point}")
            
            if potential_source:
                source = potential_source
                break  # Stop once we find a valid source

        # Identify destination node (last point inside a bbox)
        for point in reversed(path):
            potential_destination = None
            for obj, data in objects.items():
                if is_inside_bbox(point, data["bbox"]):
                    potential_destination = obj
                    #print(f"Potential Destination: {potential_destination} for point {point}")
    
                    if potential_destination != source:  
                        destination = potential_destination
                        break  # Stop after finding a valid destination
            if destination:
                destination = potential_destination
                break  # Stop once we find a valid source
         
        #print(f"Source: {source}, Destination: {destination}")
        
        # Add path from source to destination in DAG
        if source and destination and source != destination:
            #print(f"Edge: {source} -> {destination} added to DAG\n")
            if destination not in graph[source]:  # Avoid duplicate edges
                graph[source].append(destination)

        # Store the current path for future checks
        path_list.append(path)

    # Ensure all detected output nodes exist
    all_destinations = {dest for dest_list in graph.values() for dest in dest_list}
    for dest in all_destinations:
        if dest not in graph:
            graph[dest] = []  # Output node (no outgoing edges)
    
    #print(dict(graph))
    return dict(graph)  # Convert defaultdict to normal dict

if __name__ == "__main__":
        objects = {
            "A": {"bbox": (100, 50, 200, 100)},  
            "B": {"bbox": (120, 200, 220, 250)},
            "C": {"bbox": (100, 400, 200, 450)},
            "OR": {"bbox": (300, 100, 400, 200)},
            "NOT": {"bbox": (300, 400, 350, 450)},
            "Y1": {"bbox": (500, 150, 600, 200)},
            "Y2": {"bbox": (500, 400, 600, 450)},
            "Y3": {"bbox": (600, 550, 700, 600)},
            "SUM": {"bbox": (650, 250, 750, 350)},
            "Y4": {"bbox": (800, 275, 900, 325)}
        }
        lines = [
            [(200, 75), (305, 150)],
            [(220, 225), (260, 225)],
            [(260, 225), (260, 175)],
            [(260, 175), (305, 175)],
            [(400, 150), (500, 150)],  
            [(200, 425), (300, 425)], 
            [(350, 425), (500, 425)], 
            [(350, 425), (485, 270)], 
            [(485, 270), (670, 270)],
            [(280, 425), (280, 575)],
            [(280, 575), (600, 575)],
            [(425, 425), (425, 150)],
            [(425, 150), (650, 312)],
            [(530, 575), (530, 600)],
            [(530, 600), (730, 600)],
            [(730, 600), (730, 450)],
            [(730, 450), (650, 450)],
            [(650, 450), (650, 300)],
            [(750, 300), (800, 300)] 
        ]
        connections = create_logical_paths(lines)
        '''
        connections = [
            [(200, 75), (305, 150)],   # A -> OR (straight)
            [(220, 225), (260, 225), (260, 175), (305, 175)], # B -> OR (2 corners)
            [(400, 150), (500, 150)],  # OR -> Y1 (straight)
            [(200, 425), (300, 425)], # C -> NOT (straight)
            [(350, 425), (500, 425)], # NOT -> Y2 (straight)
            [(350, 425), (485, 270), (670, 270)], #NOT -> SUM (1 corner)
            [(280, 425), (280, 575), (600, 575)], # C -> Y3 (1 corner)
            [(425, 425), (425, 150), (650, 312)],  # Y2 -> SUM (two corners)
            [(530, 575), (530, 600), (730, 600), (730, 450), (650, 450), (650, 300), (650, 300)],  # Y3 -> SUM (one corner)
            [(750, 300), (800, 300)]  # SUM -> Y4 (straight)
        ]'''
        
        dag = build_dag(objects, connections) 
        print(dag)
