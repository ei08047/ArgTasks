"""
Node: abstract Node class
"""

class Node:
    
    def __init__(self, nodeInfo, edges):
        self.nodeInfo= nodeInfo
        self.edges= edges
        
    def __str__(self):
        outputString= "Show Node:" + '\n' + "--- NodeInfo:" + '\n' + str(self.nodeInfo) + '\n' + "--- Edges" + '\n'
        
        for e in self.edges:
            outputString= outputString + str(e) + '\n'
        
        return outputString + '\n'
    
