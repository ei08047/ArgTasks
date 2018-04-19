"""
Token: typical Token produced by postagger tool -> adapted according to CitiusTagger output
"""

class Token:
    
    def __init__(self, content, lemma, tags):
        # original set of words
        self.content= content
        
        # canonical form of a set of words (lemma)
        self.lemma= lemma
        
        # tags produced as output from postagger
        self.tags= tags
    
    
    def getTokenCategory(self):
        return self.tags
    
    def getLemma(self):
        return self.lemma
    
    def getContent(self):
        return self.content
        
    
    def __str__(self):
        return self.content + " | " + self.lemma + " | " + self.tags