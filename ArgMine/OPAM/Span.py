import re


class Span:

    def __init__(self, start, end):
        self.start = int(start)
        self.end = int(end)

    def match(self, other):
        if self.start < other.start and self.end < other.start:
            return False
        elif self.start > other.start and other.end < self.start:
            return False
        else:
            return True

    def fuzzy_match(self,other):
        if self.start < other.start and self.end < other.start:
            return -1
        elif self.start > other.start and other.end < self.start:
            return -1
        elif self.start <= other.start:
            return (self.end - other.start) /  (self.end - self.start)
        elif self.start >= other.start:
            return (other.end - self.start) /  (self.end - self.start)

    #  TODO: return a span
    def locate(self, pattern, text):
        return re.findall(pattern,text)

    def __str__(self):
        return str(str(self.start) + ' : ' + str(self.end))
