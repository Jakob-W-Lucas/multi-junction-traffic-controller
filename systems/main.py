from road_network.graph import Node, Edge, Graph, NodeType
import road_network.graph_tools as gt

if __name__ == "__main__":
    
    # North road (pointing down toward south)
    n_stop1 = Node([1.25,  2.0], [0.0, -1.0], NodeType.STOP, "n_stop1")
    n_stop2 = Node([0.75,  2.0], [0.0, -1.0], NodeType.STOP, "n_stop2")
    n_stop3 = Node([0.25,  2.0], [0.0, -1.0], NodeType.STOP, "n_stop3")

    n_end1  = Node([-1.25,  2.0], [0.0, 1.0], NodeType.END, "n_end1")
    n_end2  = Node([-0.75,  2.0], [0.0, 1.0], NodeType.END, "n_end2")
    n_end3  = Node([-0.25,  2.0], [0.0, 1.0], NodeType.END, "n_end3")

    # South road (pointing up toward north)
    s_stop1 = Node([-1.25, -2.0], [0.0, 1.0], NodeType.STOP, "s_stop1")
    s_stop2 = Node([-0.75, -2.0], [0.0, 1.0], NodeType.STOP, "s_stop2")
    s_stop3 = Node([-0.25, -2.0], [0.0, 1.0], NodeType.STOP, "s_stop3")

    s_end1  = Node([1.25, -2.0], [0.0, -1.0], NodeType.END, "s_end1")
    s_end2  = Node([0.75, -2.0], [0.0, -1.0], NodeType.END, "s_end2")
    s_end3  = Node([0.25, -2.0], [0.0, -1.0], NodeType.END, "s_end3")

    # West road (pointing right toward east)
    w_stop1 = Node([-2.0, 0.25], [ 1.0, 0.0], NodeType.STOP, "w_stop1")
    w_stop2 = Node([-2.0, 0.75], [ 1.0, 0.0], NodeType.STOP, "w_stop2")

    w_end1  = Node([-2.0, -0.25], [ -1.0, 0.0], NodeType.END, "w_end1")
    w_end2  = Node([-2.0, -0.75], [ -1.0, 0.0], NodeType.END, "w_end2")

    # East road (pointing left toward west)
    e_stop1 = Node([ 2.0, -0.25], [-1.0, 0.0], NodeType.STOP, "e_stop1")
    e_stop2 = Node([ 2.0, -0.75], [-1.0, 0.0], NodeType.STOP, "e_stop2")

    e_end1  = Node([ 2.0,  0.25], [1.0, 0.0], NodeType.END, "e_end1")
    e_end2  = Node([ 2.0,  0.75], [1.0, 0.0], NodeType.END, "e_end2")

    
    edges = []

    # North road (top -> bottom, stop to end)
    edges.extend([Edge(n_stop1, s_end1), Edge(n_stop2, s_end2), Edge(n_stop3, s_end3)])

    # South road (bottom -> top, stop to end) 
    edges.extend([Edge(s_stop1, n_end1), Edge(s_stop2, n_end2), Edge(s_stop3, n_end3)])
    
    # East road (right -> left, stop to end)
    edges.extend([Edge(e_stop1, w_end1), Edge(e_stop2,w_end2)])

    # West road (left -> right, stop to end)
    edges.extend([Edge(w_stop1, e_end1), Edge(w_stop2, e_end2)])
    
    # Turns
    
    # North road turns:
    edges.extend([Edge(n_stop1, e_end2), Edge(n_stop1, e_end1), Edge(n_stop3, w_end1), Edge(n_stop3, w_end2)])
    
    # South road turns:
    edges.extend([Edge(s_stop1, w_end2), Edge(s_stop1, w_end1), Edge(s_stop3, e_end1), Edge(s_stop3, e_end2)])
    
    # East road turns:
    edges.extend([Edge(e_stop1, n_end3), Edge(e_stop1, n_end2), Edge(e_stop1, n_end1), Edge(e_stop2, s_end3), Edge(e_stop2, s_end2), Edge(e_stop2, s_end1)])
    
    # West road turns:
    edges.extend([Edge(w_stop1, s_end3), Edge(w_stop1, s_end2), Edge(w_stop1, s_end1), Edge(w_stop2, n_end3), Edge(w_stop2, n_end2), Edge(w_stop2, n_end1)])
    
    all_nodes = (
        n_stop1, n_stop2, n_stop3,
        n_end1, n_end2, n_end3,
        s_stop1, s_stop2, s_stop3,
        s_end1, s_end2, s_end3,
        w_stop1, w_stop2,
        w_end1, w_end2,
        e_stop1, e_stop2,
        e_end1, e_end2
    )
    
    graph = Graph(all_nodes, edges)
    
    graph.plot()
