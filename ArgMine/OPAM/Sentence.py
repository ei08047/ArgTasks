import itertools
import re
from OPAM.SubjectivityClues import SubjectivityClues
from OPAM.Verb import Verb

class Sentence:
    newid = itertools.count()
    sc = SubjectivityClues()
    ukp_verb = Verb()

    def __init__(self, span, text):
        self.id = next(Sentence.newid)
        self.span = span
        self.text = text
        self.annotation_list = []

    def add_annotation(self,annotation):
        self.annotation_list.append(annotation)

    def __str__(self):
        return str('{' + str(self.span.start) + '||' + self.text + '||'+str(self.span.end) + '}')

    def get_annotations(self):
        return self.annotation_list

    def lower(self):
        return self.text.lower()

    def view_annotation_list(self):
        for ann in self.annotation_list:
            print('ann::',ann)

    def view_gold_annotation(self):
        for ann in self.annotation_list:
            if(ann.is_arg()):
                print('ann::',ann)

    def get_sentence_type(self):
        arg_annotations = [a for a in self.annotation_list if (re.match('Premise|Claim|MajorClaim', a.ann_type))  ]
        if(arg_annotations != [] ):
            if(len(arg_annotations) > 1 ):
                all_premises = [a for a in arg_annotations if (re.match('Premise', a.ann_type))]
                all_claims = [a for a in arg_annotations if (re.match('Claim|MajorClaim', a.ann_type))]
                if(len(all_premises) == len(arg_annotations)):
                    return 'Premise'
                elif(len(all_claims) == len(arg_annotations)):
                    return 'Claim'
                else:
                    return 'Premise&Claim'
            else:
                annotation = arg_annotations.pop()
                if (re.match('Claim|MajorClaim', annotation.ann_type)):
                    return 'Claim'
                elif (re.match('Premise', annotation.ann_type)):
                    return 'Premise'
        else:
            return 'None'

    def isArgumentative(self):
        if([a for a in self.annotation_list if (re.match('Premise|Claim|MajorClaim', a.ann_type))  ] != [] ):
            return 'Arg'
        else:
            return 'Non-Arg'

    def is_claim(self):
        if([a for a in self.annotation_list if (re.match('Claim|MajorClaim', a.ann_type))  ] != [] ):
            return 'Claim'
        else:
            return 'Other'

    def is_premise(self):
        if([a for a in self.annotation_list if (re.match('Premise', a.ann_type))  ] != [] ):
            return 'Premise'
        else:
            return 'Other'

    def opinion_finder_features(self):
        features = dict()
        feat = [(anno.ann_type,anno.text) for anno in self.get_annotations() if (re.match('opinion_finder', anno.ann_type))]
        if (len(feat) == 1):
            (ann_type,text) = feat.pop()
            features[ann_type] = text
        else:
            print('not found: ',self.id,self.text)
        return features

    def arguing_features(self):
        features = {'assessment':False, 'authority':False, 'causation':False, 'conditionals':False, 'contrast':False,'difficulty':False, 'doubt':False, 'emphasis':False, 'generalization':False, 'inconsistency':False,'inyourshoes':False, 'necessity':False, 'possibility':False, 'priority':False,'rhetoricalquestion':False, 'structure':False, 'wants':False}
        alist = [anno.ann_type for anno in self.get_annotations() if (re.match('^is\w+', anno.ann_type))]
        for a in alist:
            if a in features.keys():
                features[a] = True
        return features

    def verb_features(self):
        features = {'has_mental': False, 'has_communication':False}
        features['has_mental'] = Sentence.ukp_verb.is_sentence_with_mental_verb(self.text)
        features['has_communication'] = Sentence.ukp_verb.is_sentence_with_communication_verb(self.text)
        return features


        # TODO: use prior-polarity sentence patterns

    def subjectivity_features(self):
        feature = {'strongsubj':False, 'weaksubj':False} # , 'strongsubj_count': 0, 'weaksubj_count':0, 'subj_pattern':''
        feature['strongsubj'] = Sentence.sc.is_sentence_with_strong_subjectivity(self.text)
        #feature['strongsubj_count'] = Sentence.sc.count_strong(self.text)
        feature['weaksubj'] = Sentence.sc.is_sentence_with_weak_subjectivity(self.text)
        #feature['weaksubj_count'] = Sentence.sc.count_weak(self.text)
        #feature['subj_pattern'] = Sentence.sc.subj_pattern(self.text)
        return feature

    def get_features_in_list(self,l):
        features = dict()
        for feat in l:
            method_to_call = getattr(self, feat)
            result = method_to_call()
            for k,v in result.items():
                features[k] = v
        return features

    def get_all_features(self):
        opin = self.opinion_finder_features()
        arg = self.arguing_features()
        subj = self.subjectivity_features()
        verb = self.verb_features()
        features = dict()
        for ok,ov in opin.items():
            features[ok]=ov
        for ak,av in arg.items():
            features[ak]=av
        for sk,sv in subj.items():
            features[sk]=sv
        for vk,vv in verb.items():
            features[vk]=vv
        return features
