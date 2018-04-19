from Span import Span
from re import match


class Annotation:

    def __init__(self, ann_type, ann_start, ann_end, ann_text):
        self.ann_type = ann_type
        self.span = Span(ann_start,ann_end)
        self.text = ann_text

    def __str__(self):
        return str(self.__dict__ )

    def is_arg(self):
        return match('Claim|MajorClaim|Premise',self.ann_type)

    def is_opinion(self):
        return match('opinion_finder',self.ann_type)

    def is_arguing_lexicon(self):
        return match('is\w+',self.ann_type)

    def is_subjectivity_clue(self):
        return match('weaksubj|strongsubj',self.ann_type)