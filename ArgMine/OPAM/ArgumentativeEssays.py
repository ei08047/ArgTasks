import os.path
import re
from Sentence import Sentence
from Span import Span
from nltk import sent_tokenize
from ArguingLexicon import ArguingLexicon
from SubjectivityClues import SubjectivityClues
from Annotation import Annotation
from Experiment import Experiment
import  itertools
from random import shuffle

class Essay:
    al = ArguingLexicon()
    sc = SubjectivityClues()

    def __init__(self,title,body):
        self.title = ''.join(str(e) for e in title)
        self.body =''.join(str(e) for e in body)
        self.annotations = []
        self.sentences = []

    def add_annotation(self,anns):
        if(isinstance(anns,dict)):
            for an_id,an_val in anns.items():
                self.annotations.append(an_val)
        elif(isinstance(anns,list)):
            for an in anns:
                self.annotations.append(an)

    def get_arguing_annotations(self):
        return [annotation for annotation in self.annotations if re.match('Claim|Premise|MajorClaim' ,annotation.ann_type )]

    def get_opinion_finder_annotations(self):
        return [annotation for annotation in self.annotations if annotation.ann_type == 'opinion_finder']

    def get_arguing_lexicon_annotations(self):
        return [ann for ann in self.annotations if (re.match('^is\w+',ann.ann_type))]

    def get_subjectivity_clues_annotations(self):
        return [ann for ann in self.annotations if (re.match('weaksubj|strongsubj', ann.ann_type))]

    def find_subjectivity_lexicon(self):
        ret = []
        for sent in self.sentences:
            sent_span = sent.span
            #print('SPAN::',sent_span)
            sent = sent.text.lower()
            #print(sent)
            b = Essay.sc.analyse_sentence(sent)
            #print('find_subjectivity_lexicon::',b)
            for (word,subj_type,pol) in b: #TODO: use polarity
                index = sent.find(word)
                subjectivity_clue_annotation = Annotation(subj_type, sent_span.start + index, sent_span.start + index + len(word), word )
                ret.append(subjectivity_clue_annotation)
        return ret

    def find_arguing_lexicon(self):
        ret=[]
        for sent in self.sentences:
            sent_span = sent.span
            #print('SPAN::',sent_span)
            sent = sent.text.lower()
            #print(sent)
            b = Essay.al.SentenceFragment(sent)
            for k,list in b.items():
                for v in list:
                    #print('key:',k)
                    #print('value:', v)
                    index = sent.find(v)
                    #print('begin:',sent_span.start+index,' | end:' ,sent_span.start+index+len(v) )
                    a_l_annotations = Annotation(k, sent_span.start+index, sent_span.start+index+len(v), v)
                    ret.append(a_l_annotations)
        return ret

    def tokenize_body(self):
        offset = len(self.title)
        sentences = sent_tokenize(self.body) ## only needed in python 2 (.decode('utf-8'))
        for sent in sentences:
            # print('current offset::',offset, '|| next offset::', offset+len(sent)  )
            next_offset = offset + len(sent) + 1
            s = Span(offset, next_offset)
            s = Sentence(s, sent)
            self.sentences.append(s)
            offset = next_offset

    def assign_annotations_to_sentences(self):
        for sentence in self.sentences:
            #print('_______{}_______'.format(sentence))
            sent_span = sentence.span
            for annotation in self.annotations:
                if( annotation.is_arg()):
                    if sent_span.fuzzy_match(annotation.span) > 0.7:
                        sentence.add_annotation(annotation)
                if(annotation.is_opinion()):
                    if sent_span.fuzzy_match(annotation.span) > 0.7:
                        sentence.add_annotation(annotation)
                elif annotation.is_arguing_lexicon():
                    if(annotation.span.match(sent_span)):
                        sentence.add_annotation(annotation)
                        #print(annotation.ann_type,annotation.span, ' matched sentence', sent_span)
                elif annotation.is_subjectivity_clue():
                    if(annotation.span.match(sent_span)):
                        #print(annotation.ann_type,annotation.span, ' matched sentence', sent_span)
                        sentence.add_annotation(annotation)

    def __str__(self):
        return self.title

class ArgumentativeEssays:
    essay_name_pattern = '(essay\d+).txt$'
    corpora_path = os.path.expanduser('~/nltk_data/corpora')
    path_to_opinion_finder = 'opinionfinderv2.0/'
    opinion_finder_database = 'database/docs/'
    argumentative_essays_input = 'argumentative_essays/'
    annotation_output = '_auto_anns/sent_subj.txt'

    def __init__(self):
        self.doclist = []
        self.essay_dict = {}

    def read_doc_list(self):  # read doc_list
        doclist_file = str(ArgumentativeEssays.path_to_opinion_finder + ArgumentativeEssays.opinion_finder_database + ArgumentativeEssays.argumentative_essays_input)
        for o in os.listdir(doclist_file):
            if(re.match(ArgumentativeEssays.essay_name_pattern, o)):
                o = re.findall(ArgumentativeEssays.essay_name_pattern, o).pop()
                self.doclist.append(o)
        print('doc_list', doclist_file, ' with {} documents'.format(len(self.doclist)))

    def read_all_raw(self):
        for doc_id in self.doclist:
            (title, body) = self.read_raw(doc_id)  # read raw
            essay = Essay(title,body)
            self.add_essay_entry(doc_id,essay)

    def read_raw(self, doc_id):
        argumentative_essays_corpora_path = str(ArgumentativeEssays.corpora_path + '/argumentative_essays/')
        enc = 'utf-8'
        path_to_doc_text = str(argumentative_essays_corpora_path + doc_id + '.txt')
        if (os.path.exists(path_to_doc_text)):
            with open(path_to_doc_text ,mode='r', encoding="utf-8") as raw:
                title = raw.readline()
                body = raw.readlines()
                return (title, body)

    def add_essay_entry(self, essay_id, essay_entry):
        self.essay_dict[essay_id] = essay_entry

    def get_essay_by_id(self, id):
        return self.essay_dict[id]

    def read_arg_annotation(self, path):
        capture_anns_pattern = '(T\d+).(Claim|Premise|MajorClaim).(\d+).(\d+).(.+)'
        with open(path, 'r') as anns:
            temp_anns = {}#  ann_id : Annotation
            lines = anns.readlines()
            for line in lines:
                if (re.match(capture_anns_pattern, line)):
                    (id, ann, start, end, span) = re.findall(capture_anns_pattern, line).pop()
                    temp_anns[id] = Annotation(ann, start, end, span)
        return temp_anns

    def produce_all_arguing_lexicon_annotations(self):
        for essay_id,essay in self.essay_dict.items():
            list = essay.find_arguing_lexicon()
            essay.add_annotation(list)

    def produce_all_subjectivity_lexicon_annotations(self):
        for essay in self.essay_dict.values():
            list = essay.find_subjectivity_lexicon()
            essay.add_annotation(list)

    def produce_sentences_in_essays(self):
        for essay in self.essay_dict.values():
            essay.tokenize_body()

    def read_all_arg_annotations(self):
        argumentative_essays_corpora_path = str(ArgumentativeEssays.corpora_path + '/argumentative_essays/')
        for doc_id in self.doclist:
            path_to_doc_ann = str(argumentative_essays_corpora_path + doc_id + '.ann')
            anns = self.read_arg_annotation(path_to_doc_ann)
            essay = self.get_essay_by_id(doc_id)
            essay.add_annotation(anns)

    def read_all_opinion_finder_annotations(self):
        capture_pattern = 'argumentative_essays_(essay\d{1,3}).txt_(\d*)_(\d*)\t(subj|obj)\n'
        for doc in self.doclist:
            path_to_annotation = str(
                ArgumentativeEssays.path_to_opinion_finder + ArgumentativeEssays.opinion_finder_database + ArgumentativeEssays.argumentative_essays_input + doc + '.txt' + ArgumentativeEssays.annotation_output)
            if (os.path.exists(path_to_annotation)):
                with open(path_to_annotation, 'r') as anno_file:
                    lines = anno_file.readlines()
                    temp_list = []
                    for line in lines:
                        (doc_id, start, end, classification) = re.findall(capture_pattern, line).pop()
                        #print(doc_id, start, end, classification)
                        opinion_finder_annotation = Annotation('opinion_finder',start, end, classification)
                        temp_list.append(opinion_finder_annotation)
                    self.essay_dict[doc_id].add_annotation(temp_list)
                    # print(doc_id,start,end,classification)
            else:
                print('could not find: ',path_to_annotation)

    def get_all_sentences(self):
        sentences = [sentence for essay in self.essay_dict.values() for sentence in essay.sentences ]
        print('get_all_sentences {} '.format(len(sentences)))
        return sentences

    def get_labeled_sentences_by_target(self, sentences, target):
        labeled_sentences = []
        for (essay_id, essay) in self.essay_dict.items():
            for sentence in essay.sentences:
                claim_target = sentence.is_claim()
                premise_target = sentence.is_premise()
                arg_target = sentence.isArgumentative()
                for t in target:
                    if(t == 'Claim' and t == claim_target):
                        labeled_sentences.append((sentence.id, claim_target))
                    elif(t == 'Premise' and t == premise_target):
                        labeled_sentences.append((sentence.id, premise_target))
                    elif(t == 'Arg' and t == arg_target):
                        labeled_sentences.append((sentence.id, arg_target))
                    elif(t == 'Non-Arg' and t == arg_target):
                        labeled_sentences.append((sentence.id, arg_target))
                        #TODO: solve Other issue
                    # 'Claim', 'Premise', 'Arg' ,  'Non-Arg', 'Other'
        return labeled_sentences

    # Arg, Non-Arg
    def get_labeled_sentences(self,sentences):
        labeled_sentences = []
        for (essay_id, essay) in ae_corpora.essay_dict.items():
            for sentence in essay.sentences:
                labeled_sentences.append((sentence.id, sentence.isArgumentative()))
        return labeled_sentences

    ## premise , claim, none
    def get_labeled_sentences2(self, sentences):
        labeled_sentences = []
        for (essay_id, essay) in ae_corpora.essay_dict.items():
            for sentence in essay.sentences:
                labeled_sentences.append((sentence.id, sentence.get_sentence_type()))
        return labeled_sentences

    ## claim , other
    def get_labeled_sentences3(self,sentences):
        labeled_sentences = []
        for (essay_id, essay) in ae_corpora.essay_dict.items():
            for sentence in essay.sentences:
                labeled_sentences.append((sentence.id, sentence.is_claim()))
        return labeled_sentences

    ## Premise , Other
    def get_labeled_sentences4(self,sentences):
        labeled_sentences = []
        for (essay_id, essay) in ae_corpora.essay_dict.items():
            for sentence in essay.sentences:
                labeled_sentences.append((sentence.id, sentence.is_premise()))
        return labeled_sentences


    def assign_features(self):
        for essay in self.essay_dict.values():
            essay.assign_annotations_to_sentences()

    def get_sentence_by_id(self, id):
        for doc in self.essay_dict.values():
            for s in doc.sentences:
                if (s.id == int(id)):
                    return s


test = False
if(test):
    ae_corpora = ArgumentativeEssays()
    ae_corpora.read_doc_list()
    ae_corpora.read_all_raw()
    ae_corpora.read_all_arg_annotations()

    ae_corpora.read_all_opinion_finder_annotations()
    ae_corpora.produce_sentences_in_essays()
    ae_corpora.produce_all_arguing_lexicon_annotations()
    ae_corpora.produce_all_subjectivity_lexicon_annotations()
    ae_corpora.assign_features()
    sentences = ae_corpora.get_all_sentences()


full_list_of_features = ['arguing_features', 'opinion_finder_features', 'verb_features', 'subjectivity_features']
all_comb = [list(comb) for i in range(1, len(full_list_of_features)+1) for comb in itertools.combinations(full_list_of_features, i)]

##################################################################
ex1= False
ex1_description = "first experience"
if(ex1):
    for comb in all_comb:
        # Arg, Non-Arg
        print('starting Arg vs Non-Arg Experience')
        labeled_sentences = ae_corpora.get_labeled_sentences(sentences)
        epx1 = Experiment(ae_corpora,comb,labeled_sentences,['Arg', 'Non-Arg'])
        epx1.one_vs_other()

    for comb in all_comb:
        # Claim vs Other
        print('starting Claim vs Other Experience')
        labeled_sentences = ae_corpora.get_labeled_sentences3(sentences)
        epx1 = Experiment(ae_corpora,comb,labeled_sentences,['Claim', 'Other'])
        epx1.one_vs_other()

    for comb in all_comb:
        # Premise , Other
        print('starting Premise vs Other Experience')
        labeled_sentences = ae_corpora.get_labeled_sentences4(sentences)
        epx1 = Experiment(ae_corpora,comb,labeled_sentences,['Premise', 'Other'])
        epx1.one_vs_other()


    for comb in all_comb:
        # multi-class
        print('starting Claim, Premise and None Experience')
        labeled_sentences = ae_corpora.get_labeled_sentences2(sentences)
        epx1 = Experiment(ae_corpora,comb,labeled_sentences,['Claim', 'Premise', 'None'])
        epx1.run_multi_class()
##################################################################
ex2= False
ex2_description = "second experience: ( Claim vs Non-Arg | Premise vs Non-Arg | Claim vs Premise) "
if(ex2):
    print(ex2_description)
    #   Claim vs Non-Arg
    for comb in all_comb:

        target = ('Claim', 'Non-Arg')
        print('starting Claim vs Non-Arg Experience')
        labeled_sentences = ae_corpora.get_labeled_sentences_by_target(sentences,target)
        num_claims = len([sent_id for (sent_id, sent_label) in labeled_sentences if sent_label == 'Claim'])
        num_non_arg = len([sent_id for (sent_id, sent_label) in labeled_sentences if sent_label == 'Non-Arg'])
        print('claims {} non-Args {}'.format(num_claims,num_non_arg))

        epx2 = Experiment(ae_corpora,comb,labeled_sentences,[target[0], target[1]])
        epx2.one_vs_other()
    #  Premise vs Non-Arg
    for comb in all_comb:
        target = ('Premise', 'Non-Arg')
        print('starting Premise vs Non-Arg Experience')
        labeled_sentences = ae_corpora.get_labeled_sentences_by_target(sentences,target)
        num_premises = len([sent_id for (sent_id, sent_label) in labeled_sentences if sent_label == 'Premise'])
        num_non_arg = len([sent_id for (sent_id, sent_label) in labeled_sentences if sent_label == 'Non-Arg'])
        print('premises {} non-Args {}'.format(num_premises,num_non_arg))
        epx2 = Experiment(ae_corpora,comb,labeled_sentences,[target[0], target[1]])
        epx2.one_vs_other()
    #  Claim vs Premise
    target = ('Claim', 'Premise')
    print('starting Claim vs Premise Experience')
    labeled_sentences = ae_corpora.get_labeled_sentences_by_target(sentences, target)
    premises = [(sent_id, sent_label) for (sent_id, sent_label) in labeled_sentences if sent_label == 'Premise']
    claims = [(sent_id, sent_label) for (sent_id, sent_label) in labeled_sentences if sent_label == 'Claim']
    num_claims = len(claims)
    num_premises = len(premises)
    print('claims: {} | premises: {}'.format(num_claims, num_premises))
    claims.extend(premises[:num_claims])
    shuffle(claims)
    balanced_labeled_sentences = claims
    for comb in all_comb:
        # Claim vs Premise
        epx2 = Experiment(ae_corpora, comb, balanced_labeled_sentences, [target[0], target[1]])
        epx2.one_vs_other()
##################################################################
ex3 = False
if(ex3):
    ex3_description = "third experiment : ( Claim vs Premise ->  Target )"
    target = ('Claim', 'Premise', 'Arg')

    print('starting Baseline Experience')
    labeled_sentences = ae_corpora.get_labeled_sentences_by_target(sentences, target)
    for comb in all_comb:
        epx3 = Experiment(ae_corpora, comb, labeled_sentences, [target[0], target[1],target[2]])
    #    epx3.test_pipeline()




    def show_essay(essay_id):
        essay1 = ae_corpora.get_essay_by_id(essay_id)
        print('__________________{}_____________________________'.format(essay_id))
        print('title: ',str(essay1))
        print('________________________________________________________')
        print('body : {} sentences',format(len(essay1.sentences)))
        for s in essay1.sentences:
            print(s)
        print('________________________________________________________')

        essay1_arg = essay1.get_arguing_annotations()
        print(len(essay1_arg), ' ARG ANNOTATIONS')
        for arg_ann in essay1_arg:
            print(str(arg_ann))

        print('________________________________________________________')
        essay1_opin = essay1.get_opinion_finder_annotations()
        print(len(essay1_opin), ' OPINION_FINDER FEATURE')
        for opin_ann in essay1_opin:
            print(str(opin_ann))

        print('________________________________________________________')
        essay1_arguing_lexicon = essay1.get_arguing_lexicon_annotations()
        print(len(essay1_arguing_lexicon), ' ARGUING LEXICON FEATURE')
        for arguing_lexicon_ann in essay1_arguing_lexicon:
            print(str(arguing_lexicon_ann))
        print('________________________________________________________')
        essay1_subjectivity_clues_annotations = essay1.get_subjectivity_clues_annotations()
        print(len(essay1_subjectivity_clues_annotations), ' SUBJECTIVITY CLUES FEATURE')
        for subjectivity_clue_ann in essay1_subjectivity_clues_annotations:
            print(str(subjectivity_clue_ann))

    def show_essay_sentences(essay_id):
        essay = ae_corpora.get_essay_by_id(essay_id)
        for sentence in essay.sentences:
            print(sentence)
            sentence.view_gold_annotation()

    #show_essay('essay001')
    #show_essay_sentences('essay001')










