import numpy as np
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from nav_env.obstacles.obstacles import Obstacle, Circle

class VoronoiNavigationGraph:
    """
    A simple class to create navigation graphs from Voronoi diagrams.
    """
    
    def __init__(self, obstacles, xlim, ylim):
        self.obstacles = obstacles
        self.xlim = xlim
        self.ylim = ylim
        self.vor = None
        self.graph = None
        
    def create_voronoi(self, n_points_per_obstacle=30, n_boundary_points=12):
        """Create Voronoi diagram from obstacle boundaries."""
        # Sample points from obstacle boundaries
        seed_points = []
        for obs in self.obstacles:
            boundary = obs.exterior
            distances = np.linspace(0, boundary.length, n_points_per_obstacle, endpoint=False)
            for distance in distances:
                point = boundary.interpolate(distance)
                seed_points.append([point.x, point.y])
        
        # Add boundary points
        boundary_points = self._generate_boundary_points(n_boundary_points)
        all_points = np.vstack([seed_points, boundary_points])
        
        self.vor = Voronoi(all_points)
        return self.vor
    
    def _generate_boundary_points(self, n_points_per_side):
        """Generate boundary points around the perimeter."""
        boundary_points = []
        
        # Bottom edge
        x_vals = np.linspace(self.xlim[0], self.xlim[1], n_points_per_side)
        for x in x_vals:
            boundary_points.append([x, self.ylim[0]])
        
        # Right edge (excluding corners)
        y_vals = np.linspace(self.ylim[0], self.ylim[1], n_points_per_side)[1:]
        for y in y_vals:
            boundary_points.append([self.xlim[1], y])
        
        # Top edge (excluding corners)
        x_vals = np.linspace(self.xlim[1], self.xlim[0], n_points_per_side)[1:]
        for x in x_vals:
            boundary_points.append([x, self.ylim[1]])
        
        # Left edge (excluding corners)
        y_vals = np.linspace(self.ylim[1], self.ylim[0], n_points_per_side)[1:-1]
        for y in y_vals:
            boundary_points.append([self.xlim[0], y])
        
        return boundary_points
    
    def _edge_is_safe(self, pos1, pos2, min_clearance=0.02):
        """Check if an edge between two positions is safe (doesn't intersect obstacles)."""
        line = LineString([pos1, pos2])
        
        for obs in self.obstacles:
            if obs.intersects(line):
                return False
            if obs.distance(line) < min_clearance:
                return False
        
        return True
    
    def create_navigation_graph(self, min_clearance=0.05, edge_clearance=0.02, simplify=True):
        """Create navigation graph from Voronoi vertices."""
        if self.vor is None:
            self.create_voronoi()
        
        G = nx.Graph()
        
        # Add safe vertices as nodes
        safe_vertices = {}
        for i, vertex in enumerate(self.vor.vertices):
            x, y = vertex
            # Check if vertex is within bounds
            if not (self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1]):
                continue
                
            min_dist = min(obs.distance((x, y)) for obs in self.obstacles)
            
            if min_dist >= min_clearance:
                safe_vertices[i] = (x, y)
                G.add_node(i, pos=(x, y), clearance=min_dist)
        
        # Add edges along Voronoi ridges - only if they don't intersect obstacles
        for ridge in self.vor.ridge_vertices:
            if len(ridge) == 2 and ridge[0] != -1 and ridge[1] != -1:
                v1, v2 = ridge
                if v1 in safe_vertices and v2 in safe_vertices:
                    pos1 = safe_vertices[v1]
                    pos2 = safe_vertices[v2]
                    
                    # Check if edge is safe (doesn't intersect obstacles)
                    if self._edge_is_safe(pos1, pos2, edge_clearance):
                        weight = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        G.add_edge(v1, v2, weight=weight)
        
        # Check connectivity before simplification
        print(f"Graph before simplification: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Connected components: {nx.number_connected_components(G)}")
        
        # Simplify graph while preserving connectivity
        if simplify:
            G = self._simplify_graph_safely(G, edge_clearance)
        
        print(f"Graph after simplification: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Connected components: {nx.number_connected_components(G)}")
        
        self.graph = G
        return G
    
    def _simplify_graph_safely(self, G, edge_clearance):
        """Remove leaf nodes and redundant degree-2 nodes while preserving connectivity."""
        # Remove leaf nodes iteratively
        while True:
            leaf_nodes = [node for node, degree in G.degree() if degree == 1]
            if not leaf_nodes:
                break
            G.remove_nodes_from(leaf_nodes)
        
        # Remove redundant degree-2 nodes more carefully
        while True:
            degree_2_nodes = [node for node, degree in G.degree() if degree == 2]
            nodes_removed = 0
            
            for node in degree_2_nodes:
                neighbors = list(G.neighbors(node))
                if len(neighbors) != 2:
                    continue
                
                neighbor1, neighbor2 = neighbors
                pos1 = G.nodes[neighbor1]['pos']
                pos2 = G.nodes[neighbor2]['pos']
                
                # Check if direct connection is safe and doesn't disconnect the graph
                if self._edge_is_safe(pos1, pos2, edge_clearance):
                    # Test connectivity: temporarily remove node and see if graph stays connected
                    G_temp = G.copy()
                    G_temp.remove_node(node)
                    
                    # Only remove if it doesn't increase the number of connected components
                    if nx.number_connected_components(G_temp) <= nx.number_connected_components(G):
                        weight = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        G.remove_node(node)
                        if not G.has_edge(neighbor1, neighbor2):
                            G.add_edge(neighbor1, neighbor2, weight=weight)
                        nodes_removed += 1
            
            if nodes_removed == 0:
                break
        
        return G
    
    def find_junction_paths(self, min_junction_degree=3):
        """Find paths between junctions through degree-2 nodes."""
        if self.graph is None:
            self.create_navigation_graph()
        
        # Work only with the largest connected component
        if nx.number_connected_components(self.graph) > 1:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            working_graph = self.graph.subgraph(largest_cc).copy()
            print(f"Working with largest connected component: {working_graph.number_of_nodes()} nodes")
        else:
            working_graph = self.graph
        
        junctions = set(node for node, degree in working_graph.degree() if degree >= min_junction_degree)
        degree_2_nodes = set(node for node, degree in working_graph.degree() if degree == 2)
        
        print(f"Found {len(junctions)} junctions and {len(degree_2_nodes)} degree-2 nodes")
        
        paths = []
        
        for start_junction in junctions:
            for neighbor in working_graph.neighbors(start_junction):
                if neighbor in junctions:
                    continue  # Skip direct junction connections
                elif neighbor in degree_2_nodes:
                    path = self._trace_degree_2_path(start_junction, neighbor, junctions, degree_2_nodes, working_graph)
                    if path and len(path) >= 3:
                        paths.append(path)
        
        # Remove duplicate paths
        unique_paths = []
        seen_pairs = set()
        for path in paths:
            start, end = path[0], path[-1]
            if (start, end) not in seen_pairs and (end, start) not in seen_pairs:
                unique_paths.append(path)
                seen_pairs.add((start, end))
        
        return unique_paths
    
    def _trace_degree_2_path(self, start_junction, first_degree_2_node, junctions, degree_2_nodes, graph):
        """Trace path through degree-2 nodes to next junction."""
        path = [start_junction, first_degree_2_node]
        current_node = first_degree_2_node
        visited = {start_junction, first_degree_2_node}
        
        while True:
            neighbors = [n for n in graph.neighbors(current_node) if n not in visited]
            
            if len(neighbors) == 0:
                return None  # Dead end
            elif len(neighbors) == 1:
                next_node = neighbors[0]
                if next_node in junctions:
                    path.append(next_node)
                    return path
                elif next_node in degree_2_nodes:
                    path.append(next_node)
                    visited.add(next_node)
                    current_node = next_node
                else:
                    return None  # Invalid path
            else:
                return None  # Multiple neighbors shouldn't happen for degree-2
    
    def create_corridor_lines(self, strategy='collision_aware'):
        """Create straight line corridors between junctions."""
        paths = self.find_junction_paths()
        all_segments = []
        
        for path in paths:
            segments = self._create_line_segments(path, strategy)
            all_segments.extend(segments)
        
        return all_segments, paths
    
    def _create_line_segments(self, path, strategy='collision_aware'):
        """Create line segments for a path."""
        if len(path) < 3:
            return []
        
        positions = [self.graph.nodes[node]['pos'] for node in path]
        
        if strategy == 'single':
            return [[positions[0], positions[-1]]]
        
        elif strategy == 'collision_aware':
            # Start with single segment and refine if it intersects obstacles
            segments = [[positions[0], positions[-1]]]
            
            for iteration in range(3):  # Max 3 refinements
                new_segments = []
                
                for segment in segments:
                    start_pos, end_pos = segment
                    
                    if self._edge_is_safe(start_pos, end_pos):
                        new_segments.append(segment)
                    else:
                        # Split segment using path waypoints
                        mid_idx = len(positions) // 2
                        mid_pos = positions[mid_idx]
                        new_segments.extend([[start_pos, mid_pos], [mid_pos, end_pos]])
                
                segments = new_segments
                
                # Check if all segments are now safe
                all_safe = all(self._edge_is_safe(seg[0], seg[1]) for seg in segments)
                
                if all_safe:
                    break
            
            return segments
        
        else:  # adaptive
            num_intermediate = len(path) - 2
            if num_intermediate <= 2:
                return [[positions[0], positions[-1]]]
            elif num_intermediate <= 5:
                mid_idx = len(positions) // 2
                return [[positions[0], positions[mid_idx]], [positions[mid_idx], positions[-1]]]
            else:
                third_1 = len(positions) // 3
                third_2 = 2 * len(positions) // 3
                return [
                    [positions[0], positions[third_1]], 
                    [positions[third_1], positions[third_2]],
                    [positions[third_2], positions[-1]]
                ]
    
    def plot(self, show_voronoi=True, show_graph=True, show_corridors=True, strategy='collision_aware'):
        """Plot everything together."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot Voronoi diagram
        if show_voronoi and self.vor is not None:
            voronoi_plot_2d(self.vor, ax=ax, show_vertices=False, 
                           line_colors='lightgray', line_width=0.5, show_points=False)
        
        # Plot obstacles
        for obs in self.obstacles:
            obs.plot(ax=ax)
        
        # Plot navigation graph
        if show_graph and self.graph is not None:
            pos = nx.get_node_attributes(self.graph, 'pos')
            
            # Classify nodes
            junctions = [node for node, degree in self.graph.degree() if degree >= 3]
            degree_2_nodes = [node for node, degree in self.graph.degree() if degree == 2]
            other_nodes = [node for node in self.graph.nodes() if node not in junctions and node not in degree_2_nodes]
            
            # Plot edges
            nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color='gray', width=1, alpha=0.5)
            
            # Plot nodes
            if other_nodes:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=other_nodes, ax=ax, 
                                     node_color='lightblue', node_size=20, alpha=0.7)
            if degree_2_nodes:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=degree_2_nodes, ax=ax, 
                                     node_color='green', node_size=30, alpha=0.8)
            if junctions:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=junctions, ax=ax,
                                     node_color='red', node_size=100, alpha=0.9)
        
        # Plot corridor lines
        if show_corridors:
            segments, paths = self.create_corridor_lines(strategy)
            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(paths))))
            
            for i, path in enumerate(paths):
                path_segments = self._create_line_segments(path, strategy)
                
                # Plot original path
                path_positions = [self.graph.nodes[node]['pos'] for node in path]
                path_x = [pos[0] for pos in path_positions]
                path_y = [pos[1] for pos in path_positions]
                ax.plot(path_x, path_y, 'o-', color=colors[i], alpha=0.3, markersize=2, linewidth=1)
                
                # Plot corridor segments
                for j, segment in enumerate(path_segments):
                    start_pos, end_pos = segment
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                           color=colors[i], linewidth=4, alpha=0.9,
                           label=f'Corridor {i+1}' if j == 0 else "")
        
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal')
        ax.set_title(f'Voronoi Navigation Graph with Corridors\n'
                    f'Strategy: {strategy}')
        
        if show_corridors and len(paths) > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax

def test():
    """Test the simplified navigation graph."""
    # Create obstacles
    o1 = Obstacle(xy=[(-2, -2), (2, -2), (2, 2), (-2, 2)]).rotate(30).translate(1, 3)
    o2 = Circle(0, 0, 3).translate(-3, -2)
    o3 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(-30).translate(3, -3)
    o4 = Circle(0, 0, 3).translate(7, 2)
    o5 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(-30).translate(-5, -8)
    obstacles = [o1, o2, o3, o4, o5]
    
    # Create navigation graph
    nav = VoronoiNavigationGraph(obstacles, xlim=(-10, 12), ylim=(-12, 7))
    
    # Build everything
    nav.create_voronoi(n_points_per_obstacle=30)
    nav.create_navigation_graph(min_clearance=0.0, edge_clearance=0.0, simplify=True)
    
    # Print statistics
    print(f"Voronoi vertices: {len(nav.vor.vertices)}")
    if nav.graph:
        print(f"Graph nodes: {nav.graph.number_of_nodes()}")
        print(f"Graph edges: {nav.graph.number_of_edges()}")
        
        paths = nav.find_junction_paths()
        print(f"Junction paths found: {len(paths)}")
        for i, path in enumerate(paths):
            print(f"  Path {i+1}: {len(path)-2} intermediate nodes")
    
    # Plot results
    nav.plot(show_voronoi=True, show_graph=True, show_corridors=True, strategy='collision_aware')

if __name__ == "__main__":
    test()