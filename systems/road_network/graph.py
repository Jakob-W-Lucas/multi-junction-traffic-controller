import matplotlib.pyplot as plt
import numpy as np

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
    
    def __init__(self, position: tuple[float, float], node_type: NodeType):
        
        self.position = position
        self.node_type = node_type
    
class Edge:
    """
    Graph edge for connecting nodes and storing cell data
    """
    def __init__(self, start: Node, stop: Node, cell_length: float, line_args = None):
        """
        
        Stores the start and end node and propagates a curve over a range.
        
        Args:
            start (Node): Start node of the edge, can be a spawning node
            stop (Node): Stop node of the edge, can be intersecting node or an end node
            cell_length (float): Total length for each cell on the edge
        """
        
        _line_args = {
            "x_range": (0, 20), # Range of x values to plot over
            "eq": lambda x: 10,           # Equation as function or in lambda notation,
                                # e.g. eq = def parabola(x): return x**2 OR eq = lambda x: x**2. 
                                # Defaults to lambda x: 10 (straight edge with y = 10).
            "res_pul": 5,       # The resolution per unit length of the edge. Defaults to 5.
        }
        if line_args is not None:
            for key in line_args.keys():
                _line_args[key] = line_args[key]
        
        # Create an array of x values
        res = np.abs(_line_args["x_range"][0] - _line_args["x_range"][1]) * _line_args["res_pul"]
        x = np.linspace(_line_args["x_range"][0], _line_args["x_range"][1], res)
        # Apply the equation to the x array
        y = _line_args["eq"](x)
        
        # Create the cells along the edge
        dx = np.diff(x)
        dy = np.diff(y)
        # Calculate length of each segment and sum them
        length = np.sum(np.sqrt(dx**2, dy**2))
        cell_count = np.floor(length / cell_length)
        
        self.x, self.y = x, y
        self.cells = [Cell(self) for _ in cell_count]
        
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
    
    