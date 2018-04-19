"""
Edge: abstract Edge class
"""

class Edge:
    
    def __init__(self, edgeInfo, destinationNodeId):
        self.edgeInfo= edgeInfo
        self.destinationNodeId= destinationNodeId
        
    def getEdgeInfo(self):
        return self.edgeInfo
    
    def getDestinationNodeId(self):
        return self.destinationNodeId
        
    def __str__(self):
        return "Edge:" + '\n' + "--- EdgeInfo \n" + str(self.edgeInfo) + '\n' + "--- DestinationNodeId" + '\n' + "   " + str(self.destinationNodeId) + '\n'
