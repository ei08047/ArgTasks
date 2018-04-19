"""
Graph: abstract Graph class
"""

class Graph:
    
    def __init__(self, nodes):
        self.nodes= nodes
    
    def __str__(self):
        outputString =  "Show Graph: " + '\n'
        for n in self.nodes:
            outputString = outputString + str(n)
        return outputString + '\n'
