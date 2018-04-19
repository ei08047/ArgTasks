"""
ArgumentDiagram: contains the argument diagram (graph structure obtained from a particular 
text) and all extra information such as original text and respective identifiers in 
the different sources
"""

class ArgumentDiagram:
    
    def __init__(self, graph, originalText, newsId, filename):
        # set of arguments from a particular text (graph structure)
        self.graph= graph
        
        # complete and original text
        self.originalText= originalText
        
        # corresponds to the "newsId" in the database
        self.newsId= newsId
        
        # corresponding filename at 'ArgMineCorpus' folder -> created by AIFDB
        self.filename= filename
        
    def __str__(self):
        return "+++ Argument Diagram +++" + '\n' + "+ Graph:" + '\n' + str(self.graph) + '\n' + "+ Original Text:" + '\n' + str(self.originalText) + '\n' + '\n' + "+ News Id= " + str(self.newsId) + '\n' + '\n' + "+ Filename= " + str(self.filename) + '\n'
        
