from collections import defaultdict
import math

THRESHOLD_DISTANCE = 5  # pixels
VARIATION_THRESHOLD = 2  # Max allowable x/y variation to normalize

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def normalize_points(points):
    """Normalize small X and Y variations without merging distinct segments."""
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

def build_graph(line_segments):
    """Creates an adjacency list representation of the graph."""
    graph = defaultdict(set)
    for start, end in line_segments:
        graph[start].add(end)
        graph[end].add(start)
    return {key: list(value) for key, value in graph.items()}

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        """Find the representative of a node with path compression."""
        if node not in self.parent:
            self.parent[node] = node
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        """Merge two sets."""
        root1, root2 = self.find(node1), self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1

def merge_close_points(graph):
    """Merges nodes that are within THRESHOLD_DISTANCE."""
    uf = UnionFind()
    nodes = list(graph.keys())

    # Initialize all nodes in UnionFind
    for node in nodes:
        uf.parent[node] = node  

    # Merge nodes within the distance threshold
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if euclidean_distance(nodes[i], nodes[j]) <= THRESHOLD_DISTANCE:
                uf.union(nodes[i], nodes[j])

    # Rebuild graph with merged nodes
    new_graph = defaultdict(set)
    for node, neighbors in graph.items():
        root_node = uf.find(node)
        for neighbor in neighbors:
            root_neighbor = uf.find(neighbor)
            if root_neighbor != root_node:
                new_graph[root_node].add(root_neighbor)

    return {key: list(value) for key, value in new_graph.items()}

def find_connected_components(graph):
    """Finds connected components in the graph using DFS."""
    visited = set()
    paths = []

    for start_node in graph:
        if start_node not in visited:
            stack = [start_node]
            path = []
            lines = []

            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    path.append(node)
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)
                            lines.append((node, neighbor))

            # Normalize the path to smooth out minor variations
            path = normalize_points(path)

            # Preserve the order to avoid unexpected reordering
            paths.append((path, lines))

    return paths

def organize_paths(line_segments):
    """Main function to process the line segments and return organized paths."""
    graph = build_graph(line_segments)
    merged_graph = merge_close_points(graph)
    return find_connected_components(merged_graph)

# Example line segments
lines = [
    ((209, 391), (513, 391)),
((211, 257), (640, 257)),
((211, 300), (640, 300)),
((213, 347), (641, 347)),
((215, 129), (594, 129)),
((230, 632), (230, 389)),
((230, 632), (626, 632)),
((341, 192), (367, 193)),
((343, 475), (465, 475)),
((367, 154), (592, 154)),
((367, 193), (367, 154)),
((447, 551), (661, 551)),
((557, 393), (642, 393)),
((559, 476), (619, 476)),
((599, 597), (614, 599)),
((613, 585), (662, 585)),
((614, 599), (613, 585)),
((616, 460), (618, 440)),
((619, 476), (618, 440)),
((624, 612), (791, 610)),
((626, 632), (624, 612)),
((670, 141), (965, 141)),
((734, 572), (791, 572)),
((756, 295), (1103, 295)),
((762, 404), (1103, 404)),
((774, 224), (775, 299)),
((774, 224), (817, 224)),
((832, 77), (898, 77)),
((844, 590), (886, 590)),
((892, 223), (928, 223)),
((897, 116), (966, 115)),
((898, 77), (897, 116)),
((926, 169), (967, 167)),
((928, 223), (926, 169)),
((929, 591), (1104, 591)),
((1022, 142), (1101, 142))
]

# Organize paths and print results
new_paths = sorted(organize_paths(lines))

for (path, connections) in new_paths:
    print(path)
