from collections import defaultdict, deque

# Sample DAG representation as an adjacency list
'''
Input:
A-----v
       AND---->OR---->Y
B-----^         ^
   |            |
   |            |
   ----v        |
C----->XOR--->NOT

'''
graph = {
    "A": ["AND"],
    "B": ["AND", "XOR"],
    "C": ["XOR"],
    "AND": ["OR"],
    "XOR": ["NOT"],
    "NOT": ["OR"],
    "OR": ["Y1"],
    "Y1": [],
}

# Function definitions for operations
'''
function_definitions = {
    "AND": "int AND(int a, int b) { return a & b; }",
    "XOR": "int XOR(int a, int b) { return a ^ b; }",
    "OR": "int OR(int a, int b) { return a | b; }"
}'''

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
def generate_c_code(graph):
    topo_order = topological_sort(graph)
    c_code = ["#include <stdio.h>"]
    c_code.append("#include <math.h>\n")
    '''
    # Add function prototypes
    for func in function_definitions.values():
        c_code.append(func)
    '''
    c_code.append("\nint main() {")

    temp_var_count = 1
    temp_vars = {}
    output_nodes = []

    for node in topo_order:
        if node in {"A", "B", "C"}:
            #continue  # Skip input nodes
            c_code.append(f"    double {node};  //values not defined")
        elif node.startswith("Y"):
            # Store output nodes for dynamic print statements
            output_nodes.append(node)
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

# Generate and print the C code
try:
    #c_code = generate_c_code(graph, function_definitions)
    c_code = generate_c_code(graph)
    print(c_code)
except KeyError as e:
    print(f"ERROR: {e}")
