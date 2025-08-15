import sys, os, matplotlib.pyplot as plt
from nav_env.control.path import Waypoints
from nav_env.map.map import Map
from corridor_voronoi import *

NX, NY = 201, 201
center=(350000, 6.04e6)
size=(25e3, 0.02e6)
start=(350000, 6032000)
goal=(342220, 6049000)

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_path)

config_path = os.path.join('examples', 'config', 'san_juan_channel.yaml')
shore = Map(config_path)
# obstacles = shore.get_grid_cell_environment(nx=NX, ny=NY, center=center, size=size)
obstacles = shore.get_obstacle_collection_in_window_from_enc(depth=10)

xlim = (497000, 503000)
ylim = (5.380e6, 5.386e6)

# Option 1: Voronoi from centroids
vor_centroids, seed_points_centroids = build_voronoi_from_obstacles(obstacles, xlim, ylim)

# Option 2: Voronoi from boundary samples
vor_boundary, seed_points_boundary = build_voronoi_from_boundary_samples(obstacles, xlim, ylim, meters_between_points=10)
nav_graph = prune_graph(voronoi_to_navigation_graph(vor_boundary, obstacles, xlim, ylim, min_clearance=0.1))

# Plot both approaches
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))

# Plot centroid-based Voronoi
plot_voronoi_with_obstacles(vor_centroids, obstacles, seed_points_centroids, xlim, ylim, ax1)
ax1.set_title("Voronoi from Centroids")

# Plot boundary-based Voronoi
plot_voronoi_with_obstacles(vor_boundary, obstacles, seed_points_boundary, xlim, ylim, ax2)
ax2.set_title("Voronoi from Boundary Samples")
# Adaptive collision-aware strategy
plot_degree_2_straight_lines(nav_graph, vor_boundary, obstacles, xlim, ylim, 
                                strategy='collision_aware', ax=ax3)

plt.tight_layout()
plt.show()