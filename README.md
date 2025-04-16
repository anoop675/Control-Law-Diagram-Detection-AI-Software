# Control Law Diagram Detection Software for Automatic C code Generation for FADEC System, GTRE DRDO internship project

# Algorithm: 
Control Law Diagram -> Object & Line Detection -> Logical Path Formation (Connections) ->  Directed Acyclic Graph (DAG) Construction -> Logical Mapping & Topological Sort -> C code Generation
<img width="602" alt="image" src="https://github.com/user-attachments/assets/8b6bc98d-972d-4e9c-a48e-48b6b8c8475e" />


# Input:
Accept a logic diagram image (input_image).

# Preprocessing:
Perform image enhancement:
Apply contrast adjustment and denoising filters (e.g., Gaussian blur, median filtering) using OpenCV or similar libraries.
Convert the image to grayscale or binary for further processing.
Detect and eliminate irrelevant artifacts (e.g., gridlines, smudges).

# Symbol Detection:
Use a pre-trained computer vision model (e.g., CNN or Vision Transformer) to:
Detect components such as logic gates, connectors, and annotations.
Classify symbols (e.g., AND, OR, NOT gates) and store their properties (type, position, size).
Apply Optical Character Recognition (OCR) to extract text labels.

# Graph Creation:
Identify relationships between symbols (e.g., lines connecting gates).
Construct a directed graph:
Nodes represent symbols/components.
Edges represent connections or signal flow.

# Logic Mapping:
Traverse the graph to:
Derive logical relationships (e.g., A AND B, A OR B).
Assign labels and variables to inputs, outputs, and intermediate nodes.

# Code Generation:
Translate the logical relationships into executable code:
Use a template-based generator for the target language (e.g., Python, C, ladder logic, Verilog).
Generate functions or modules corresponding to the logic gates and their connections.

# Output:
Return the generated code as a script or file.
