from collections import defaultdict, deque

#Checks if a point (x, y) is inside a bounding box (x_min, y_min, x_max, y_max)
def is_inside_bbox(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    
    is_x_within_x_bbox = x_min <= x <= x_max
    is_y_within_y_bbox = y_min <= y <= y_max
    
    return is_x_within_x_bbox and is_y_within_y_bbox

# Build DAG using objects and connections
def build_dag(objects, connections):
    graph = {}

    for path in connections:
        source, destination = None, None
    
        # Identify source node (first point inside a bbox)
        for point in path:
            for obj, data in objects.items():
                if is_inside_bbox(point, data["bbox"]):
                    source = obj
                    break
            if source:
                break
    
        # Identify destination node (last point inside a bbox)
        for point in reversed(path):
            for obj, data in objects.items():
                if is_inside_bbox(point, data["bbox"]):
                    destination = obj
                    break
            if destination:
                break
    
        # Add source → destination to DAG
        if source and destination and source != destination:
            graph.setdefault(source, []).append(destination)
    
    #Ensure all detected output nodes (destinations with no outgoing edges) exist
    all_destinations = {dest for dest_list in graph.values() for dest in dest_list}
    for dest in all_destinations:
        if dest not in graph:
            graph[dest] = []  # Output node (no outgoing edges)
            
    #print("Directed Acyclic Graph: \n"+str(graph))        
    return graph

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
    topo_order = topological_sort(graph)
    c_code = ["#include <stdio.h>"]
    c_code.append("#include <math.h>\n")
    '''
    # Add function prototypes
    for func in function_definitions.values():
        c_code.append(func)
    '''
    c_code.append("TODO: insert function definitions\n")
    c_code.append("int main(void) {")

    temp_var_count = 1
    temp_vars = {}
    output_nodes = []

    for node in topo_order:
        if node in {"A", "B", "C"}:
            #continue  # Skip input nodes
            c_code.append(f"    double {node};  //values not defined")
        elif node.startswith("Y"):
            output_nodes.append(node) # Store output nodes for dynamic print statements
            # Set output variable directly from its dependency
            input_nodes = [n for n in graph if node in graph[n]]
            if input_nodes:
                input_var = temp_vars[input_nodes[0]]  # Get temp variable of dependency
                c_code.append(f"    {node} = {input_var};")
            else:
                raise KeyError(f"Node '{node}' has no computed value.")
        else:
            # Generate a temporary variable for the current node
            inputs = [var for var in graph if node in graph[var]]
            input_vars = [temp_vars[i] if i in temp_vars else i for i in inputs]
            temp_var = f"temp{temp_var_count}"
            temp_vars[node] = temp_var
            temp_var_count += 1
            c_code.append(f"    double {temp_var} = {node}({', '.join(input_vars)});")

    # Generate dynamic printf statements for output nodes
    for output in output_nodes:
        c_code.append(f'    printf("{output} = %f\\n", {output});')

    c_code.append('    return 0;')
    c_code.append('}')

    return "\n".join(c_code)

if __name__ == "__main__":
    try:
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
        objects = {
            "A": {"bbox": (100, 50, 200, 100)},  
            "B": {"bbox": (120, 200, 220, 250)},
            "C": {"bbox": (100, 400, 200, 450)},
            "OR": {"bbox": (300, 100, 400, 200)},
            "NOT": {"bbox": (300, 400, 350, 450)},
            "Y1": {"bbox": (500, 150, 600, 200)},
            "Y2": {"bbox": (500, 400, 600, 450)}
        }
        connections = [
            [(200, 75), (305, 150)],   # A -> OR (straight)
            [(220, 225), (260, 225), (260, 175), (305, 175)], # B -> OR (2 corners)
            [(400, 150), (500, 150)],  # OR -> Y1 (straight)
            [(200, 425), (300, 425)], # C -> NOT (straight)
            [(350, 425), (500, 425)] # NOT -> Y2 (straight)
        ]
        
        dag = build_dag(objects, connections) 
        #c_code = generate_c_code(dag, function_definitions)
        c_code = generate_c_code(dag)
        print(c_code)
    except KeyError as e:
        print(f"ERROR: {e}")
