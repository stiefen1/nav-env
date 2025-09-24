import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import networkx as nx

# Example: Create a Voronoi graph from a set of points

# 1. Define some points
points = np.array([
    [0, 0],
    [2, 1],
    [1, 3],
    [3, 4],
    [4, 0],
    [5, 2]
])

# 2. Compute the Voronoi diagram
vor = Voronoi(points)

# 3. Plot the Voronoi diagram
fig, ax = plt.subplots(figsize=(8, 6))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=2, point_size=30)
ax.set_title("Simple Voronoi Diagram from Points")
ax.legend()
plt.tight_layout()
plt.show()

# 4. Build a graph from Voronoi vertices and ridges
G = nx.Graph()
for i, vertex in enumerate(vor.vertices):
    G.add_node(i, pos=vertex)

for ridge in vor.ridge_vertices:
    if len(ridge) == 2 and ridge[0] != -1 and ridge[1] != -1:
        v1, v2 = ridge
        G.add_edge(v1, v2)

# 5. Plot the Voronoi graph structure
fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, ax=ax, node_color='cyan', edge_color='gray', with_labels=True, node_size=100)
ax.set_title("Voronoi Graph Structure (Vertices and Edges)")
plt.tight_layout()
plt.show()