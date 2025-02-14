from collections import defaultdict, deque

# Sample DAG representation as an adjacency list
graph = {
    "A": ["AND"],
    "B": ["AND", "XOR"],
    "C": ["XOR"],
    "AND": ["OR"],
    "XOR": ["OR", "Y2"],
    "OR": ["Y1"],
    "Y1": [],
    "Y2": []
}

# Function definitions for operations
function_definitions = {
    "AND": "int AND(int a, int b) { return a & b; }",
    "XOR": "int XOR(int a, int b) { return a ^ b; }",
    "OR": "int OR(int a, int b) { return a | b; }"
}

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

    return topo_order

# Generate C code from topological order
def generate_c_code(graph, function_definitions):
    topo_order = topological_sort(graph)
    c_code = ["#include <stdio.h>"]
    
    # Add function prototypes
    for func in function_definitions.values():
        c_code.append(func)
    
    c_code.append("\nint main() {")
    c_code.append("    int A = 1, B = 0, C = 1;")
    c_code.append("    int Y1, Y2;")

    temp_var_count = 1
    temp_vars = {}

    for node in topo_order:
        if node in {"A", "B", "C"}:
            continue  # Skip input nodes
        elif node.startswith("Y"):
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
            temp_var = f"temp_{temp_var_count}"
            temp_vars[node] = temp_var
            temp_var_count += 1
            c_code.append(f"    int {temp_var} = {node}({', '.join(input_vars)});")

    c_code.append('    printf("Y1 = %d\\n", Y1);')
    c_code.append('    printf("Y2 = %d\\n", Y2);')
    c_code.append('    return 0;')
    c_code.append('}')

    return "\n".join(c_code)

# Generate and print the C code
try:
    c_code = generate_c_code(graph, function_definitions)
    print(c_code)
except KeyError as e:
    print(f"ERROR: {e}")
