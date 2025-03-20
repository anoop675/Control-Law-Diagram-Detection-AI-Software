from collections import defaultdict
import math

THRESHOLD_DISTANCE = 5  # pixels
VARIATION_THRESHOLD = 2  # Max allowable x/y variation to normalize

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def normalize_points(points):
    """Normalize small X and Y variations within the threshold."""
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
    graph = defaultdict(set)
    for start, end in line_segments:
        graph[start].add(end)
        graph[end].add(start)
    return {key: list(value) for key, value in graph.items()}

class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    def union(self, node1, node2):
        root1, root2 = self.find(node1), self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1

def merge_close_points(graph):
    uf = UnionFind()
    nodes = list(graph.keys())
    for node in nodes:
        uf.parent[node] = node
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if euclidean_distance(nodes[i], nodes[j]) <= THRESHOLD_DISTANCE:
                uf.union(nodes[i], nodes[j])
    new_graph = defaultdict(set)
    for node, neighbors in graph.items():
        root_node = uf.find(node)
        for neighbor in neighbors:
            root_neighbor = uf.find(neighbor)
            if root_neighbor != root_node:
                new_graph[root_node].add(root_neighbor)
    return {key: list(value) for key, value in new_graph.items()}

def find_connected_components(graph):
    visited, paths = set(), []
    for start_node in graph:
        if start_node not in visited:
            stack, path, lines = [start_node], [], []
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    path.append(node)
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)
                            lines.append((node, neighbor))
            path = normalize_points(path)
            path.sort(key=lambda p: (p[1], p[0]))  # Sort by Y, then X
            paths.append((path, lines))
    return paths

def organize_paths(line_segments):
    graph = build_graph(line_segments)
    merged_graph = merge_close_points(graph)
    return find_connected_components(merged_graph)

lines = [
    ((213, 347), (641, 347)),
    ((762, 404), (1103, 404)),
    ((215, 129), (594, 129)),
    ((211, 257), (640, 257)),
    ((211, 300), (640, 300)),
    ((557, 393), (642, 393)),
    ((209, 391), (513, 391)),
    ((230, 632), (626, 632)),
    ((230, 632), (230, 389)),
    ((626, 632), (624, 612)),
    ((756, 295), (1103, 295)),
    ((670, 141), (965, 141)),
    ((367, 154), (592, 154)),
    ((367, 193), (367, 154)),
    ((1022, 142), (1101, 142)),
    ((929, 591), (1104, 591)),
    ((447, 551), (661, 551)),
    ((343, 475), (465, 475)),
    ((624, 612), (791, 610)),
    ((559, 476), (619, 476)),
    ((619, 476), (618, 440)),
    ((774, 224), (817, 224)),
    ((774, 224), (775, 299)),
    ((892, 223), (928, 223)),
    ((928, 223), (926, 169)),
    ((832, 77), (898, 77)),
    ((898, 77), (897, 116)),
    ((613, 585), (662, 585)),
    ((614, 599), (613, 585)),
    ((897, 116), (966, 115)),
    ((734, 572), (791, 572)),
    ((844, 590), (886, 590)),
    ((341, 192), (367, 193)),
    ((926, 169), (967, 167)),
    ((616, 460), (618, 440)),
    ((599, 597), (614, 599))
]

new_paths = organize_paths(lines)
for idx, (path, connections) in enumerate(new_paths, start=1):
    print(f"Path {idx}: {path}")
    print("  Line Segments:")
    for segment in connections:
        print(f"    {segment}")
    print()
