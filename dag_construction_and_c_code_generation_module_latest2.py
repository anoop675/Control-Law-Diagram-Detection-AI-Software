from collections import defaultdict, deque

#Checks if a point (x, y) is inside a bounding box (x_min, y_min, x_max, y_max)
def is_inside_bbox(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    
    is_x_within_x_bbox = x_min <= x <= x_max
    is_y_within_y_bbox = y_min <= y <= y_max
    
    return is_x_within_x_bbox and is_y_within_y_bbox

def get_existing_source(point, objects, connections, path_list):
    source = None
    coinciding_point = point  # First point in the path
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

# Topological Sort to get the order of function calls
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in graph if in_degree[node] == 0])
    topo_order = []

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
    c_code = ["#include <stdio.h>", "#include <stdlib.h>", "#include <float.h>", "#include <stdbool.h>", "#include <string.h>", "#include <math.h>\n"]
    c_code.append("// TODO: function definitions here")
    '''   
    function_nodes = {node for node in graph if node not in {"A", "B", "C"} and not node.startswith("Y")}

    # Generate function prototypes
    for func in function_nodes:
        if func == "OR":
            c_code.append("double OR(double, double);")
        elif func == "NOT":
            c_code.append("double NOT(double);")
        elif func == "SUM":
            c_code.append("double SUM(double, double);")
    '''
    c_code.append("\nint main(void) {")
    
    temp_var_count = 1
    temp_vars = {}  # To map logical functions to temporary variables
    output_nodes = []

    for node in topo_order:
        if node in {"A", "B", "C"}:  # Input nodes
            c_code.append(f"    double {node};  // Input variable")
            continue

        inputs = [var for var in graph if node in graph[var]]  # Get input nodes
        input_vars = [temp_vars.get(i, i) for i in inputs]  # Replace with temp vars if needed

        if node.startswith("Y"):  # Output nodes
            if len(input_vars) == 1:
                c_code.append(f"    double {node} = {input_vars[0]};")
            else:
                c_code.append(f"    double {node};  // Undefined behavior detected")
            output_nodes.append(node)
        else:  # Intermediate computations (e.g., OR, NOT)
            temp_var = f"temp{temp_var_count}"
            temp_vars[node] = temp_var
            temp_var_count += 1
            c_code.append(f"    double {temp_var} = {node}({', '.join(input_vars)});")

    # Print statements for output
    for output in output_nodes:
        c_code.append(f'    printf("{output} = %f\\n", {output});')

    c_code.append("\n    return 0;")
    c_code.append("}")

    return "\n".join(c_code)

if __name__ == "__main__":
    #try:
        '''
        objects = { # format: (x_min, y_min, x_max, y_max)
            "A": {"bbox": (1,1, 3,2)},  
            "B": {"bbox": (1,3, 3,4)},
            "OR": {"bbox": (5,1, 7,3)},
            "Y": {"bbox": (9,2, 11,3)}
        }
         connections = [
            [(3,1.5), (4,1.5), (5.5,1.5)],  # A -> OR (via intermediate point)
            [(3,3.5), (4,3.5), (4,2.5), (5,3)],  # B -> OR (with a corner)
            [(7,2), (8,2), (9,2)]  # OR -> Y (straight)
        ]
        
        connections = [
            [(3,1.5), (5.5,1.5)],  # A -> OR (via intermediate point)
            [(3,3.5), (4,3.5), (4,2.5), (5,3)],  # B -> OR (with 2 corners)
            [(7,2), (9,2)]  # OR -> Y (straight)
        ]
        '''
        # refer image.png in repo's main dir
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
        connections = [
            [(200, 75), (305, 150)],   # A -> OR (straight)
            [(220, 225), (260, 225), (260, 175), (305, 175)], # B -> OR (2 corners)
            [(400, 150), (500, 150)],  # OR -> Y1 (straight)
            [(200, 425), (300, 425)], # C -> NOT (straight)
            [(350, 425), (500, 425)], # NOT -> Y2 (straight)
            [(350, 425), (485, 270), (670, 270)], #NOT -> SUM (1 corner)
            [(280, 425), (280, 575), (600, 575)], # C -> Y3 (1 corner)
            #[(600, 175), (650, 275)],  # Y1 -> SUM (straight)
            [(425, 425), (425, 150), (650, 312)],  # Y2 -> SUM (two corners)
            [(530, 575), (530, 600), (730, 600), (730, 450), (650, 450), (650, 300), (650, 300)],  # Y3 -> SUM (one corner)
            [(750, 300), (800, 300)]  # SUM -> Y4 (straight)
        ]
        
        dag = build_dag(objects, connections) 
        #c_code = generate_c_code(dag, function_definitions)
        c_code = generate_c_code(dag)
        print(c_code)
    #except KeyError as e:
        #print(f"ERROR: {e}")
