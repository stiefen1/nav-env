from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
import matplotlib.pyplot as plt
from nav_env.obstacles.obstacles import Obstacle, Circle

def test() -> None:
    o1 = Obstacle(xy=[(-2, -2), (2, -2), (2, 2), (-2, 2)]).rotate(30).translate(0, 4)
    o2 = Circle(0, 0, 3).translate(-3, -2)
    o3 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(-30).translate(3, -3)
    o4 = Circle(0, 0, 3).translate(7, 2)
    o5 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(-30).translate(-5, -8)
    o6 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(150).translate(7, -7)
    obstacles = [o1, o2, o3, o4, o5]#, o6]

    xlim = (-10, 12)
    ylim = (-12, 10)
    
    # Option 1: Voronoi from centroids
    vor_centroids, seed_points_centroids = build_voronoi_from_obstacles(obstacles, xlim, ylim)
    
    # Option 2: Voronoi from boundary samples
    vor_boundary, seed_points_boundary = build_voronoi_from_boundary_samples(obstacles, xlim, ylim, meters_between_points=0.2)
    nav_graph:nx.Graph = prune_graph(voronoi_to_navigation_graph(vor_boundary, obstacles, xlim, ylim, min_clearance=0.1))

    # Plot both approaches
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))
    
    # Plot centroid-based Voronoi
    plot_voronoi_with_obstacles(vor_centroids, obstacles, seed_points_centroids, xlim, ylim, ax1)
    ax1.set_title("Voronoi from Centroids")
    
    # Plot boundary-based Voronoi
    plot_voronoi_with_obstacles(vor_boundary, obstacles, seed_points_boundary, xlim, ylim, ax2)
    ax2.set_title("Voronoi from Boundary Samples")

    # Show different strategies for straight line fitting
    # plot_degree_2_straight_lines(nav_graph, vor_boundary, obstacles, xlim, ylim, 
    #                              strategy='single', ax=ax3)
    
    # plot_degree_2_straight_lines(nav_graph, vor_boundary, obstacles, xlim, ylim, 
    #                              strategy='adaptive', ax=ax3)
    
    # plot_degree_2_straight_lines(nav_graph, vor_boundary, obstacles, xlim, ylim, 
    #                              strategy='adaptive', ax=ax3)
    # Adaptive collision-aware strategy
    _, degree_2_path = plot_degree_2_straight_lines(nav_graph, vor_boundary, obstacles, xlim, ylim, 
                                 strategy='collision_aware', ax=ax3)
    # plot_voronoi_graph(nav_graph, vor_centroids, obstacles, xlim, ylim, ax3, "navigation")
    # plot_high_degree_nodes(nav_graph, vor_boundary, obstacles, xlim, ylim, ax=ax3)


    plt.tight_layout()
    plt.show()

def build_voronoi_from_obstacles(obstacles:list[Obstacle], xlim, ylim):
    """
    Build Voronoi diagram using obstacle centroids as seed points.
    """
    # Extract centroids from obstacles
    seed_points = []
    for obs in obstacles:
        centroid = obs.centroid
        seed_points.append([*centroid])
    
    seed_points = np.array(seed_points)
    
    # Add boundary points to prevent infinite regions
    boundary_points = [
        [xlim[0], ylim[0]], [xlim[1], ylim[0]], 
        [xlim[1], ylim[1]], [xlim[0], ylim[1]],
        [xlim[0], (ylim[0]+ylim[1])/2], [xlim[1], (ylim[0]+ylim[1])/2],
        [(xlim[0]+xlim[1])/2, ylim[0]], [(xlim[0]+xlim[1])/2, ylim[1]]
    ]
    
    all_points = np.vstack([seed_points, boundary_points])
    
    # Create Voronoi diagram
    vor = Voronoi(all_points)
    
    return vor, seed_points

def sample_points_from_obstacles(obstacles:list[Obstacle], meters_between_points=50):
    """
    Sample multiple points from each obstacle's boundary.
    """
    all_points = []
    
    for obs in obstacles:
        boundary = obs.exterior
        n_points = int(boundary.length // meters_between_points)
        # Sample points along the boundary
        distances = np.linspace(0, boundary.length, n_points, endpoint=False)
        
        for distance in distances:
            point = boundary.interpolate(distance)
            all_points.append([point.x, point.y])
    
    return np.array(all_points)

def generate_boundary_points(xlim, ylim, meters_between_points=50):
    """
    Generate boundary points around the perimeter.
    
    Args:
        xlim: (min_x, max_x) tuple
        ylim: (min_y, max_y) tuple  
        n_points_per_side: Number of points per boundary side
    
    Returns:
        List of [x, y] boundary points
    """
    boundary_points = []
    
    n_points_x = int((xlim[1] - xlim[0]) // meters_between_points)
    n_points_y = int((ylim[1] - ylim[0]) // meters_between_points)
    # Bottom edge (left to right)
    x_vals = np.linspace(xlim[0], xlim[1], n_points_x)
    for x in x_vals:
        boundary_points.append([x, ylim[0]])
    
    # Right edge (bottom to top, excluding corners to avoid duplicates)
    y_vals = np.linspace(ylim[0], ylim[1], n_points_y)[1:]
    for y in y_vals:
        boundary_points.append([xlim[1], y])
    
    # Top edge (right to left, excluding corners)
    x_vals = np.linspace(xlim[1], xlim[0], n_points_x)[1:]
    for x in x_vals:
        boundary_points.append([x, ylim[1]])
    
    # Left edge (top to bottom, excluding corners)
    y_vals = np.linspace(ylim[1], ylim[0], n_points_y)[1:-1]
    for y in y_vals:
        boundary_points.append([xlim[0], y])
    
    return boundary_points

def build_voronoi_from_boundary_samples(obstacles:list[Obstacle], xlim, ylim, meters_between_points=50):
    """
    Build Voronoi diagram using sampled boundary points.
    """
    seed_points = sample_points_from_obstacles(obstacles, meters_between_points=meters_between_points)
    
    # Add boundary points
    boundary_points = generate_boundary_points(xlim, ylim, meters_between_points=meters_between_points)
    
    all_points = np.vstack([seed_points, boundary_points])
    vor = Voronoi(all_points)
    
    return vor, seed_points

def plot_voronoi_with_obstacles(vor, obstacles:list[Obstacle], seed_points, xlim, ylim, ax=None):
    """
    Plot Voronoi diagram with obstacles overlaid.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', 
                     line_width=1, point_size=4)
    
    # Plot obstacles
    for obs in obstacles:
        obs.plot(ax=ax, alpha=0.7)
    
    # Plot seed points
    ax.scatter(seed_points[:, 0], seed_points[:, 1], c='black', s=20, zorder=5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    
    return ax

def voronoi_to_navigation_graph(vor, obstacles:list[Obstacle], xlim, ylim, min_clearance=0.5):
    """
    Create a navigation graph where nodes are safe Voronoi vertices
    and edges represent collision-free paths.
    """
    G = nx.Graph()
    
    # Add only vertices that are far enough from obstacles
    safe_vertices = {}
    for i, vertex in enumerate(vor.vertices):
        x, y = vertex
        # if not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]):
        #     continue
            
        # Check clearance from all obstacles
        min_dist = min(obs.distance((x, y)) for obs in obstacles)
        
        if min_dist >= min_clearance:
            safe_vertices[i] = (x, y)
            G.add_node(i, pos=(x, y), clearance=min_dist)
    
    # Add edges between safe vertices along Voronoi ridges
    for ridge in vor.ridge_vertices:
        if len(ridge) == 2 and ridge[0] != -1 and ridge[1] != -1:
            v1, v2 = ridge
            if v1 in safe_vertices and v2 in safe_vertices:
                pos1 = safe_vertices[v1]
                pos2 = safe_vertices[v2]
                weight = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                G.add_edge(v1, v2, weight=weight)
    
    return G

def plot_voronoi_graph(G, vor, obstacles:list[Obstacle], xlim, ylim, ax=None, graph_type="ridge"):
    """
    Plot Voronoi diagram with the corresponding graph overlay.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Voronoi diagram (lightly)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='lightblue', 
                     line_width=0.5, point_size=2, show_points=False)
    
    # Plot obstacles
    for obs in obstacles:
        obs.plot(ax=ax)
    
    # Plot graph
    pos = nx.get_node_attributes(G, 'pos')
    
    if graph_type == "dual":
        # Dual graph: nodes at obstacle centroids
        nx.draw(G, pos, ax=ax, node_color='yellow', node_size=100, 
                edge_color='red', width=2, alpha=0.8)
    else:
        # Ridge/navigation graph: nodes at Voronoi vertices
        nx.draw(G, pos, ax=ax, node_color='green', node_size=30, 
                edge_color='blue', width=2, alpha=0.8)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'{graph_type.title()} Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
    
    return ax

def remove_leaf_nodes(G):
    """
    Remove all leaf nodes (nodes with degree 1) and their edges.
    Continues until no more leaf nodes exist.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Modified graph with leaf nodes removed
    """
    G_copy = G.copy()
    
    while True:
        # Find all nodes with degree 1 (leaf nodes)
        leaf_nodes = [node for node, degree in G_copy.degree() if degree == 1]
        
        if not leaf_nodes:
            # No more leaf nodes to remove
            break
        
        # Remove all leaf nodes
        G_copy.remove_nodes_from(leaf_nodes)
        print(f"Removed {len(leaf_nodes)} leaf nodes: {leaf_nodes}")
    
    return G_copy

def remove_isolated_nodes(G):
    """
    Remove all isolated nodes (nodes with degree 0).
    
    Args:
        G: NetworkX graph
    
    Returns:
        Modified graph with isolated nodes removed
    """
    G_copy = G.copy()
    isolated_nodes = list(nx.isolates(G_copy))
    G_copy.remove_nodes_from(isolated_nodes)
    
    if isolated_nodes:
        print(f"Removed {len(isolated_nodes)} isolated nodes")
    
    return G_copy

def prune_graph(G):
    """
    Remove both leaf nodes and isolated nodes from the graph.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Pruned graph
    """
    G_pruned = remove_leaf_nodes(G)
    G_pruned = remove_isolated_nodes(G_pruned)
    
    return G_pruned

def find_high_degree_nodes(G, min_degree=3):
    """
    Find all nodes with degree >= min_degree.
    
    Args:
        G: NetworkX graph
        min_degree: Minimum degree threshold (default: 3)
    
    Returns:
        List of nodes with degree >= min_degree
    """
    high_degree_nodes = [node for node, degree in G.degree() if degree >= min_degree]
    return high_degree_nodes

def analyze_node_degrees(G):
    """
    Analyze and print statistics about node degrees.
    
    Args:
        G: NetworkX graph
    """
    degrees = dict(G.degree())
    
    print(f"Graph statistics:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    
    # Count nodes by degree
    degree_counts = {}
    for degree in degrees.values():
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
    
    print(f"\nNode degree distribution:")
    for degree in sorted(degree_counts.keys()):
        print(f"  Degree {degree}: {degree_counts[degree]} nodes")
    
    # Find high-degree nodes
    high_degree_nodes = find_high_degree_nodes(G, min_degree=3)
    print(f"\nNodes with degree ≥ 3: {len(high_degree_nodes)}")
    if high_degree_nodes:
        print(f"  Node IDs: {high_degree_nodes}")
        for node in high_degree_nodes:
            print(f"    Node {node}: degree {degrees[node]}, pos {G.nodes[node].get('pos', 'N/A')}")

def plot_high_degree_nodes(G, vor, obstacles, xlim, ylim, min_degree=3, ax=None):
    """
    Plot the graph with high-degree nodes highlighted.
    
    Args:
        G: NetworkX graph
        vor: Voronoi diagram
        obstacles: List of obstacles
        xlim, ylim: Plot limits
        min_degree: Minimum degree to highlight
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Voronoi diagram (lightly)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', 
                     line_width=0.5, point_size=2, show_points=False)
    
    # Plot obstacles
    for obs in obstacles:
        obs.plot(ax=ax)
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Identify high-degree nodes
    high_degree_nodes = find_high_degree_nodes(G, min_degree)
    regular_nodes = [node for node in G.nodes() if node not in high_degree_nodes]
    
    # Plot regular nodes and edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=1, alpha=0.6)
    
    if regular_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, ax=ax, 
                              node_color='black', node_size=20, alpha=0.7)
    
    # Highlight high-degree nodes
    if high_degree_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=high_degree_nodes, ax=ax,
                              node_color='red', node_size=100, alpha=0.9)
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'Navigation Graph\n{len(high_degree_nodes)} nodes with degree ≥ {min_degree}')
    
    return ax, high_degree_nodes

def extract_junction_graph(G, min_degree=3):
    """
    Create a simplified graph containing only high-degree nodes (junctions)
    and the paths between them.
    
    Args:
        G: NetworkX graph
        min_degree: Minimum degree to consider as junction
    
    Returns:
        Simplified graph with only junctions and paths between them
    """
    junctions = find_high_degree_nodes(G, min_degree)
    
    if len(junctions) < 2:
        print("Not enough junctions to create junction graph")
        return nx.Graph()
    
    junction_graph = nx.Graph()
    
    # Add junction nodes
    for junction in junctions:
        junction_graph.add_node(junction, **G.nodes[junction])
    
    # Find shortest paths between all pairs of junctions
    for i, start_junction in enumerate(junctions):
        for end_junction in junctions[i+1:]:
            try:
                path = nx.shortest_path(G, start_junction, end_junction, weight='weight')
                path_length = nx.shortest_path_length(G, start_junction, end_junction, weight='weight')
                
                # Add edge between junctions with path length as weight
                junction_graph.add_edge(start_junction, end_junction, 
                                      weight=path_length, path=path)
            except nx.NetworkXNoPath:
                continue
    
    return junction_graph

def find_degree_2_paths_between_junctions(G, min_degree=3):
    """
    Find paths between high-degree nodes that consist only of degree-2 nodes in between.
    
    Args:
        G: NetworkX graph
        min_degree: Minimum degree to consider as junction
    
    Returns:
        List of paths where each path goes from one junction to another through only degree-2 nodes
    """
    junctions = set(find_high_degree_nodes(G, min_degree))
    degree_2_nodes = set(node for node, degree in G.degree() if degree == 2)
    
    paths = []
    visited_junctions = set()
    
    for start_junction in junctions:
        if start_junction in visited_junctions:
            continue
            
        # Explore each neighbor of the junction
        for neighbor in G.neighbors(start_junction):
            if neighbor in junctions:
                # Direct connection to another junction - skip
                continue
            elif neighbor in degree_2_nodes:
                # Start tracing through degree-2 nodes
                path = trace_degree_2_path(G, start_junction, neighbor, junctions, degree_2_nodes)
                if path and len(path) >= 3:  # At least start_junction -> degree_2_node -> end_junction
                    paths.append(path)
                    
    # Remove duplicate paths (same path in reverse)
    unique_paths = []
    seen_pairs = set()
    
    for path in paths:
        start, end = path[0], path[-1]
        if (start, end) not in seen_pairs and (end, start) not in seen_pairs:
            unique_paths.append(path)
            seen_pairs.add((start, end))
    
    return unique_paths

def trace_degree_2_path(G, start_junction, first_degree_2_node, junctions, degree_2_nodes):
    """
    Trace a path from a junction through degree-2 nodes until reaching another junction.
    
    Args:
        G: NetworkX graph
        start_junction: Starting junction node
        first_degree_2_node: First degree-2 node in the path
        junctions: Set of junction nodes
        degree_2_nodes: Set of degree-2 nodes
    
    Returns:
        List of nodes representing the path, or None if path is invalid
    """
    path = [start_junction, first_degree_2_node]
    current_node = first_degree_2_node
    visited = {start_junction, first_degree_2_node}
    
    while True:
        neighbors = [n for n in G.neighbors(current_node) if n not in visited]
        
        if len(neighbors) == 0:
            # Dead end
            return None
        elif len(neighbors) == 1:
            next_node = neighbors[0]
            
            if next_node in junctions:
                # Reached another junction - valid path
                path.append(next_node)
                return path
            elif next_node in degree_2_nodes:
                # Continue through degree-2 nodes
                path.append(next_node)
                visited.add(next_node)
                current_node = next_node
            else:
                # Reached a node that's neither junction nor degree-2 - invalid path
                return None
        else:
            # Multiple unvisited neighbors - this shouldn't happen for degree-2 nodes
            return None

def segments_intersect_obstacles(segments, obstacles:list[Obstacle]):
    """
    Check if any line segments intersect with obstacles.
    
    Args:
        segments: List of line segments [(x1, y1), (x2, y2)]
        obstacles: List of obstacle objects
    
    Returns:
        List of booleans indicating which segments intersect obstacles
    """
    from shapely.geometry import LineString
    
    intersections = []
    for segment in segments:
        start_pos, end_pos = segment
        line = LineString([start_pos, end_pos])
        
        # Check if this line intersects any obstacle
        intersects = False
        for obs in obstacles:
            if obs.intersects(line):
                intersects = True
                break
        
        intersections.append(intersects)
    
    return intersections

def create_collision_aware_segments(G, path, obstacles, max_iterations=5):
    """
    Create straight line segments that avoid obstacles by iteratively adding waypoints.
    
    Args:
        G: NetworkX graph
        path: List of node IDs (junction -> degree-2 nodes -> junction)
        obstacles: List of obstacle objects
        max_iterations: Maximum number of refinement iterations
    
    Returns:
        List of line segments that avoid obstacles
    """
    if len(path) < 3:
        return []
    
    positions = [G.nodes[node]['pos'] for node in path]
    
    # Start with a single segment
    current_segments = [[positions[0], positions[-1]]]
    
    for iteration in range(max_iterations):
        # Check which segments intersect obstacles
        intersections = segments_intersect_obstacles(current_segments, obstacles)
        
        if not any(intersections):
            # No intersections, we're done
            break
        
        # Refine segments that intersect obstacles
        new_segments = []
        
        for i, (segment, intersects) in enumerate(zip(current_segments, intersections)):
            if not intersects:
                # Segment is fine, keep it
                new_segments.append(segment)
            else:
                # Segment intersects obstacle, split it using intermediate waypoints
                start_pos, end_pos = segment
                
                # Find the range of path positions that correspond to this segment
                start_idx = 0
                end_idx = len(positions) - 1
                
                # For refined segments, we need to find the appropriate sub-path
                if len(current_segments) > 1:
                    # Calculate which part of the path this segment corresponds to
                    segment_ratio_start = i / len(current_segments)
                    segment_ratio_end = (i + 1) / len(current_segments)
                    
                    start_idx = int(segment_ratio_start * (len(positions) - 1))
                    end_idx = int(segment_ratio_end * (len(positions) - 1))
                    
                    if end_idx >= len(positions):
                        end_idx = len(positions) - 1
                
                # Add intermediate waypoint(s) from the path
                if end_idx - start_idx > 1:
                    # Use intermediate points to split the segment
                    mid_idx = (start_idx + end_idx) // 2
                    mid_pos = positions[mid_idx]
                    
                    new_segments.append([start_pos, mid_pos])
                    new_segments.append([mid_pos, end_pos])
                else:
                    # Can't split further, keep the segment (might need different approach)
                    new_segments.append(segment)
        
        current_segments = new_segments
        print(f"  Iteration {iteration + 1}: {len(current_segments)} segments")
    
    return current_segments

def create_adaptive_collision_aware_segments(G, path, obstacles, initial_segments=1, max_iterations=6):
    """
    Create segments with adaptive initial subdivision and collision avoidance.
    
    Args:
        G: NetworkX graph
        path: List of node IDs
        obstacles: List of obstacle objects
        initial_segments: Initial number of segments to try
        max_iterations: Maximum refinement iterations
    
    Returns:
        List of collision-free line segments
    """
    if len(path) < 3:
        return []
    
    positions = [G.nodes[node]['pos'] for node in path]
    
    # Create initial segments based on path length
    num_intermediate = len(path) - 2
    if num_intermediate <= 2:
        initial_segments = 1
    elif num_intermediate <= 5:
        initial_segments = 2
    else:
        initial_segments = min(3, num_intermediate // 2)
    
    # Create initial segments
    if initial_segments == 1:
        current_segments = [[positions[0], positions[-1]]]
    else:
        segment_indices = np.linspace(0, len(positions) - 1, initial_segments + 1, dtype=int)
        current_segments = []
        for i in range(len(segment_indices) - 1):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i + 1]
            current_segments.append([positions[start_idx], positions[end_idx]])
    
    # Refine segments that intersect obstacles
    for _ in range(max_iterations):
        intersections = segments_intersect_obstacles(current_segments, obstacles)
        
        if not any(intersections):
            break
        
        new_segments = []
        
        for i, (segment, intersects) in enumerate(zip(current_segments, intersections)):
            if not intersects:
                new_segments.append(segment)
            else:               
                # Find intermediate points from the original path
                # Calculate which portion of the path this segment spans
                total_segments = len(current_segments)
                path_start_ratio = i / total_segments
                path_end_ratio = (i + 1) / total_segments
                
                path_start_idx = int(path_start_ratio * (len(positions) - 1))
                path_end_idx = int(path_end_ratio * (len(positions) - 1))
                
                if path_end_idx >= len(positions):
                    path_end_idx = len(positions) - 1
                
                # Split into multiple segments using intermediate points
                if path_end_idx - path_start_idx > 1:
                    # Create 2-3 sub-segments
                    sub_positions = positions[path_start_idx:path_end_idx + 1]
                    
                    if len(sub_positions) <= 3:
                        # Split in half
                        mid_idx = len(sub_positions) // 2
                        new_segments.append([sub_positions[0], sub_positions[mid_idx]])
                        new_segments.append([sub_positions[mid_idx], sub_positions[-1]])
                    else:
                        # Split into thirds
                        third_1 = len(sub_positions) // 3
                        third_2 = 2 * len(sub_positions) // 3
                        new_segments.append([sub_positions[0], sub_positions[third_1]])
                        new_segments.append([sub_positions[third_1], sub_positions[third_2]])
                        new_segments.append([sub_positions[third_2], sub_positions[-1]])
                else:
                    # Cannot split further
                    new_segments.append(segment)
        
        current_segments = new_segments
        
    return current_segments

def create_straight_lines_from_degree_2_path(G, path, strategy='adaptive', obstacles=None):
    """
    Create straight line segments from a path that goes through degree-2 nodes.
    
    Args:
        G: NetworkX graph
        path: List of node IDs (junction -> degree-2 nodes -> junction)
        strategy: 'single', 'adaptive', 'multi', 'collision_aware', or 'adaptive_collision_aware'
        obstacles: List of obstacle objects (required for collision-aware strategies)
    
    Returns:
        List of line segments, where each segment is [(x1, y1), (x2, y2)]
    """
    if len(path) < 3:  # Need at least junction -> degree-2 -> junction
        return []
    
    # Get positions of all nodes in the path
    positions = [G.nodes[node]['pos'] for node in path]
    
    if strategy == 'single':
        # Single line from start junction to end junction
        return [[positions[0], positions[-1]]]
    
    elif strategy == 'adaptive':
        # Adaptive based on number of intermediate nodes - MORE PRECISE
        num_intermediate = len(path) - 2  # Exclude start and end junctions
        
        if num_intermediate <= 1:
            # Very few intermediate nodes: single line
            return [[positions[0], positions[-1]]]
        elif num_intermediate <= 3:  # Changed from 5 to 3
            # Few intermediate nodes: two lines using middle point
            mid_idx = len(positions) // 2
            return [
                [positions[0], positions[mid_idx]], 
                [positions[mid_idx], positions[-1]]
            ]
        elif num_intermediate <= 6:  # Changed from direct jump to 3 segments
            # Medium: three lines
            third_1 = len(positions) // 3
            third_2 = 2 * len(positions) // 3
            return [
                [positions[0], positions[third_1]], 
                [positions[third_1], positions[third_2]],
                [positions[third_2], positions[-1]]
            ]
        else:
            # Many intermediate nodes: four or more lines
            n_segments = min(5, num_intermediate // 2)  # Max 5 segments, but adaptive
            segment_indices = np.linspace(0, len(positions) - 1, n_segments + 1, dtype=int)
            
            segments = []
            for i in range(len(segment_indices) - 1):
                start_idx = segment_indices[i]
                end_idx = segment_indices[i + 1]
                segments.append([positions[start_idx], positions[end_idx]])
            
            return segments
    
    elif strategy == 'multi':
        # Use every few intermediate nodes as waypoints
        segments = []
        segment_size = max(2, (len(path) - 2) // 3)  # Divide intermediate nodes into ~3 segments
        
        start_idx = 0
        while start_idx < len(positions) - 1:
            end_idx = min(start_idx + segment_size + 1, len(positions) - 1)
            if end_idx == len(positions) - 1 or end_idx - start_idx >= segment_size:
                segments.append([positions[start_idx], positions[end_idx]])
                start_idx = end_idx
            else:
                # Last segment too small, merge with previous
                segments[-1][1] = positions[-1]
                break
        
        return segments
    
    elif strategy == 'collision_aware':
        # Start simple and add waypoints only when segments intersect obstacles
        if obstacles is None:
            # Fallback to adaptive if no obstacles provided
            return create_straight_lines_from_degree_2_path(G, path, 'adaptive')
        
        return create_collision_aware_segments(G, path, obstacles)
    
    elif strategy == 'adaptive_collision_aware':
        # Adaptive initial segmentation with collision refinement
        if obstacles is None:
            return create_straight_lines_from_degree_2_path(G, path, 'adaptive')
        
        return create_adaptive_collision_aware_segments(G, path, obstacles)
    
    else:
        return [[positions[0], positions[-1]]]

def plot_degree_2_straight_lines(G, vor, obstacles, xlim, ylim, min_degree=3, strategy='adaptive', ax=None):
    """
    Plot straight lines between junctions connected only through degree-2 nodes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Voronoi diagram (lightly)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='lightgray', 
                     line_width=0.3, point_size=1, show_points=False)
    
    # Plot obstacles
    for obs in obstacles:
        obs.plot(ax=ax, alpha=0.7)
    
    # Get node positions and classify nodes
    pos = nx.get_node_attributes(G, 'pos')
    junctions = find_high_degree_nodes(G, min_degree)
    degree_2_nodes = [node for node, degree in G.degree() if degree == 2]
    other_nodes = [node for node in G.nodes() if node not in junctions and node not in degree_2_nodes]
    
    # Plot original graph structure lightly
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=0.5, alpha=0.3)
    
    # Plot different node types
    if other_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, ax=ax, 
                              node_color='lightblue', node_size=8, alpha=0.5)
    if degree_2_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=degree_2_nodes, ax=ax, 
                              node_color='green', node_size=15, alpha=0.7)
    if junctions:
        nx.draw_networkx_nodes(G, pos, nodelist=junctions, ax=ax,
                              node_color='red', node_size=100, alpha=0.9)
    
    # Find degree-2 paths and create straight lines
    degree_2_paths = find_degree_2_paths_between_junctions(G, min_degree)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(degree_2_paths)))
    total_segments = 0
    
    print(f"Found {len(degree_2_paths)} degree-2 paths between junctions:")
    
    for i, path in enumerate(degree_2_paths):
        print(f"  Path {i+1}: {path[0]} -> {path[1:-1]} -> {path[-1]} ({len(path)-2} degree-2 nodes)")
        
        # Create line segments with obstacle awareness if needed
        if strategy in ['collision_aware', 'adaptive_collision_aware']:
            segments = create_straight_lines_from_degree_2_path(G, path, strategy, obstacles)
        else:
            segments = create_straight_lines_from_degree_2_path(G, path, strategy)
        
        total_segments += len(segments)
        
        # Plot the original path (degree-2 nodes)
        # path_positions = [G.nodes[node]['pos'] for node in path]
        # path_x = [pos[0] for pos in path_positions]
        # path_y = [pos[1] for pos in path_positions]
        # ax.plot(path_x, path_y, 'o-', color=colors[i], alpha=0.3, markersize=3, linewidth=1)
        
        # Plot the fitted straight line segments
        for j, segment in enumerate(segments):
            start_pos, end_pos = segment
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   color=colors[i], linewidth=4, alpha=0.9,
                   label=f'Path {i+1}' if j == 0 else "")
            
            # Mark segment endpoints
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   's', color=colors[i], markersize=6, alpha=0.8)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'Straight Lines through Degree-2 Paths ({strategy})\n'
                f'{len(junctions)} junctions, {len(degree_2_paths)} degree-2 paths, {total_segments} segments')
    
    if len(degree_2_paths) > 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax, degree_2_paths



if __name__ == "__main__":
    test()