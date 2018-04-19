"""
NodeInfo: contains all information regarding a particular Node according to AIFDB
"""
class NodeInfo:
    
    def __init__(self, nodeId, text, nodeType, timestamp, scheme, schemeID):
        self.nodeId= nodeId
        self.text= text
        self.nodeType= nodeType
        self.timestamp= timestamp
        self.scheme= scheme
        self.schemeID= schemeID
        
    def getNodeId(self):
        return self.nodeId
    
    def getText(self):
        return self.text
    
    def getNodeType(self):
        return self.nodeType
    
    def getTimestamp(self):
        return self.timestamp
    
    def getScheme(self):
        return self.scheme
    
    def getSchemeID(self):
        return self.schemeID
    
    def __str__(self):
        outputString= "------ NodeInfo:" + '\n' + "          " + "nodeId= " + str(self.nodeId) + '\n' + "          " + "text= " + str(self.text) + '\n' + "          " + "nodeType= " + str(self.nodeType) + '\n' + "          " + "timestamp= " + str(self.timestamp) + '\n'
        if self.scheme is not None:
            outputString= outputString + "          " + "scheme= " + str(self.scheme) + '\n'
        return outputString
    
