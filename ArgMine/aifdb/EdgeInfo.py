"""
EdgeInfo: contains all information regarding a particular Edge according to AIFDB
"""

class EdgeInfo:
    
    def __init__(self, edgeId, formEdgeId):
        self.edgeId= edgeId
        self.formEdgeId= formEdgeId
        
    def getEdgeId(self):
        return self.edgeId
    
    def getFormEdgeId(self):
        return self.formEdgeId
        
    def __str__(self):
        return "------ EdgeInfo:" + '\n' + "          " + "edgeId= " + str(self.edgeId) + '\n' + "          " + "formEdgeId= " + str(self.formEdgeId) + '\n'
    
