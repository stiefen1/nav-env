from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

def test() -> None:
    from nav_env.obstacles.obstacles import Obstacle, Circle
    import matplotlib.pyplot as plt
    import itertools
    
    o1 = Obstacle(xy=[(-2, -2), (2, -2), (2, 2), (-2, 2)]).rotate(30).translate(1, 3)
    o2 = Circle(0, 0, 3).translate(-3, -2)
    o3 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(-30).translate(3, -3)
    o4 = Circle(0, 0, 3).translate(7, 2)
    o5 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(-30).translate(-5, -8)
    os = [o1, o2, o3, o4, o5]
    combinations = list(itertools.combinations(os, 2))

    intersections = []
    for b in np.linspace(0, 3, 15):
        for i, j in combinations:
            o1_b = i.buffer(b, join_style='mitre')
            o2_b = j.buffer(b, join_style='mitre')
            out = o1_b.exterior.intersection(o2_b.exterior)
            if out.is_empty:
                continue

            check_intersection = False
            for o in out.geoms:
                for obs in os:
                    if obs.contains(o.xy):
                        check_intersection = True
                if not check_intersection:
                    intersections.append(o.xy)

    # filtered_intersections = filter_points_kmeans(intersections, 20)
    # filtered_intersections = filter_points_dbscan(intersections, min_samples=1)
    min_distance = 2
    filtered_intersections = filter_points_spatial(intersections, min_distance=min_distance)
    G = remove_cycles_mst(build_graph_from_points(filtered_intersections, max_distance=4*min_distance))
    # G = remove_cycles_shortest_paths(build_graph_from_points(filtered_intersections, max_distance=4*min_distance))

    ax = o1.plot()
    for obs in os[1::]:
        obs.plot(ax=ax)

    plot_graph(G, ax, node_color='red', edge_color='blue', node_size=30)

    # for intersection in intersections:
    #     ax.scatter(*intersection, s=1, c='black')
    # for intersection in filtered_intersections:
    #     ax.scatter(*intersection, s=3, c='red')

    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    plt.show()


def filter_points_kmeans(points, n_clusters):
    """
    Reduce point density using K-means clustering.
    Returns cluster centroids as the filtered points.
    """
    points_array = np.array(points).squeeze(-1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(points_array)
    return kmeans.cluster_centers_.tolist()

def filter_points_dbscan(points, eps=1.0, min_samples=2):
    """
    Reduce point density using DBSCAN clustering.
    Returns one representative point per cluster.
    """
    points_array = np.array(points).squeeze(-1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points_array)
    
    filtered_points = []
    for label in set(labels):
        if label == -1:  # Skip noise points
            continue
        cluster_points = points_array[labels == label]
        # Use centroid as representative
        centroid = np.mean(cluster_points, axis=0)
        filtered_points.append(centroid.tolist())
    
    return filtered_points

def filter_points_spatial(points, min_distance=2.0):
    """
    Keep points that are at least min_distance apart.
    """
    points_array = np.array(points).squeeze(-1)
    filtered = [points_array[0]]  # Start with first point
    
    for point in points_array[1:]:
        distances = np.linalg.norm(np.array(filtered) - point, axis=1)
        if np.min(distances) >= min_distance:
            filtered.append(point)
    
    return [p.tolist() for p in filtered]

def build_graph_from_points(points, max_distance=5.0):
    """
    Build a graph from points where edges connect nearby points.
    
    Args:
        points: List of 2D points
        max_distance: Maximum distance to connect points with an edge
    
    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    
    # Add nodes (each point becomes a node)
    for i, point in enumerate(points):
        G.add_node(i, pos=point)
    
    # Add edges between nearby points
    points_array = np.array(points)
    distances = cdist(points_array, points_array)
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if distances[i, j] <= max_distance:
                G.add_edge(i, j, weight=distances[i, j])
    
    return G

def remove_cycles_mst(G):
    """
    Remove cycles by creating a minimum spanning tree.
    Keeps the graph connected with minimum total edge weight.
    """
    if G.number_of_nodes() == 0:
        return G.copy()
    
    # Create MST using edge weights (shorter distances preferred)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    return mst

def plot_graph(G, ax, node_color='red', edge_color='blue', node_size=50):
    """
    Plot the NetworkX graph on the given matplotlib axes.
    """
    # Get positions from node attributes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    nx.draw(G, pos, ax=ax, 
            node_color=node_color, 
            edge_color=edge_color,
            node_size=node_size,
            with_labels=False,
            width=1.0,
            alpha=0.7)
    
    return ax

def remove_cycles_shortest_paths(G, source_nodes=None):
    """
    Remove cycles by keeping only edges that are part of shortest paths.
    """
    if G.number_of_nodes() == 0:
        return G.copy()
    
    # If no source nodes specified, use all nodes
    if source_nodes is None:
        source_nodes = list(G.nodes())[:min(3, G.number_of_nodes())]  # Use first 3 nodes
    
    tree = nx.Graph()
    tree.add_nodes_from(G.nodes(data=True))
    
    for source in source_nodes:
        # Get shortest path tree from this source
        try:
            paths = nx.single_source_shortest_path(G, source, weight='weight')
            for target, path in paths.items():
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if not tree.has_edge(u, v):
                        tree.add_edge(u, v, weight=G[u][v]['weight'])
        except:
            continue
    
    return tree

if __name__ == "__main__":
    test()