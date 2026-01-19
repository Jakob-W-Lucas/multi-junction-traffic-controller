import matplotlib.pyplot as plt
import numpy as np

from road_network.graph_tools import bezier_curve
from sympy import symbols, solve, Eq
from enum import Enum

class NodeType(Enum):
    SPAWN = "SP"
    START = "STA", 
    STOP = "STO",
    INTERSECTION = "I",
    END = "E"

class Node:
    """
    Graph node for road intersections and connections.
    """
    
    def __init__(self, position: tuple[float, float], direction: tuple[float, float], 
                 node_type: NodeType, name: str = ""):
        self.position = position
        self.direction = direction
        self.node_type = node_type
        self.name = name

    def as_vector(self) -> tuple[tuple[float, float], tuple[float, float], NodeType]:
        """Returns position, direction, and type as a tuple for serialization or hashing."""
        return (self.position, self.direction, self.node_type)
    
class Edge:
    """
    Graph edge for connecting nodes and storing cell data
    """
    def __init__(self, start: Node, end: Node, cell_length: float = 3):
        """
        
        Stores the start and end node and propagates a curve over a range.
        """
        x, y = self.create_edge(start.as_vector(), end.as_vector())
        
        # Create the cells along the edge
        dx = np.diff(np.array(x))
        dy = np.diff(np.array(y))
        # Calculate length of each segment and sum them
        seg = np.sqrt(dx**2 + dy**2)
        length = np.sum(seg)
        cell_count = np.floor(length / cell_length)
        
        self.x, self.y = x, y
        self.cells = [Cell(self) for _ in range(int(cell_count))]
        
    def create_edge(self,
        start: tuple[tuple[float, float], tuple[float, float], NodeType], 
        end: tuple[tuple[float, float], tuple[float, float], NodeType]):
        """
        Given a start node vector and an end node vector, return the
        simple turn between the start and end and create the 
        corresponding start and end nodes and the edge connecting
        them.
        """
        
        t, s = symbols('t s')
        
        start_pos = np.asarray(start[0], dtype=float)  # [x, y]
        start_dir = np.asarray(start[1], dtype=float)  # [dx, dy]
        
        start_line = (start_pos[0] + t * start_dir[0], start_pos[1] + t * start_dir[1])
        
        end_pos = np.asarray(end[0], dtype=float)  # [x, y]
        end_dir = np.asarray(end[1], dtype=float)  # [dx, dy]
        
        end_line = (end_pos[0] + s * end_dir[0], end_pos[1] + s * end_dir[1])
        
        t_sol = solve(
            (
                Eq(start_line[0], end_line[0]), 
                Eq(start_line[1], end_line[1])
            ), 
            (t, s)
        )

        if t in t_sol and s in t_sol[t].free_symbols:
            # Collinear: t_sol[t] contains s (e.g., {t: s + 2}) -> infinite solutions
            vec_to_end = end_pos - start_pos
            dist = np.linalg.norm(vec_to_end)
            t_vals = np.linspace(0, 1, max(10, int(dist / 0.25)))  # Adaptive points by length
            points = [start_pos + t * vec_to_end for t in t_vals]
            x, y = np.array([p[0] for p in points]), np.array([p[1] for p in points])
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            
        points = [start_pos]
        
        # Get the direction from the start to the end point
        dist = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # If two lines do not intersect so we will define points along
        # their directions to act as points
        if len(t_sol) == 0:
            points.append(start_pos + start_dir * dist * 0.5)
            points.append(end_pos - end_dir * dist * 0.5)
        else:
            # Get the point of intersection of the two lines
            direction = end_pos - start_pos
            unit_vector = direction / np.linalg.norm(direction)
            print(unit_vector)
            points.append(
                [
                    start_line[0].subs(t, t_sol[t]),
                    start_line[1].subs(t, t_sol[t])
                ]
            )
            
        points.append(end_pos)
            
        x, y = bezier_curve(points)
        x_numeric, y_numeric = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        
        return x_numeric, y_numeric
        
class Cell:
    """
    Edge cell for storing data
    """
    
    def __init__(self, edge: Edge):
        
        self.edge = edge
        self.vehicle = None

class Graph:
    
    def __init__(self, nodes: tuple[Node], edges: tuple[Edge]):
        
        self.nodes = nodes
        self.edges = edges
        
    def plot(self):
        
        for node in self.nodes:
            c = "bd" if node.node_type == NodeType.STOP else "rd"
            plt.plot(node.position[0], node.position[1], c, markersize=8, markeredgecolor='k')
            plt.text(node.position[0], node.position[1] - 0.1, node.name, 
                    ha='center', va='top', fontsize=8, color='black')
            
        for edge in self.edges:
            plt.plot(edge.x, edge.y)

        plt.axis('equal')
        plt.show()