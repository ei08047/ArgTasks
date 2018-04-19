import json
import numpy as np

class Document:
    def __init__(self, parent, docleaf,output):
        self.parent = parent
        self.docleaf = docleaf
        self.output = output
        self.text_nodes = []
        self.annotation_nodes = []

    def get_doc_leaf(self):
        return self.docleaf

    def get_path(self):
        return str(self.parent + '/' + self.docleaf)

    def add_text_node(self, id, span):
        self.text_nodes.append((id,span))

    def add_annotation_node(self,attribs):
        self.annotation_nodes.append(attribs)

    def get_annotation_nodes_by_type(self, _type):
        ret = []
        for a in self.annotation_nodes:
            if a['Type'] == _type:
                ret.append(a['score'])
        return ret

    def analyse_doc(self, _type):
        num_words = len(self.text_nodes)
        scored_words =  self.get_annotation_nodes_by_type(_type)
        num_neutral = len([score for score in scored_words if score == 0])
        polarity_scores = [score for score in scored_words if score != 0]
        num_polar = len(polarity_scores)
        pos_scores = [score for score in polarity_scores if score > 0]
        neg_scores = [score for score in polarity_scores if score < 0]

        if pos_scores != []:
            max_polarity = np.max(pos_scores)
        else:
            max_polarity = None

        if neg_scores != []:
            min_polarity = np.min(neg_scores)
        else:
            min_polarity = None


        return {'num_words':num_words, 'num_neutral':num_neutral, 'num_polar':num_polar, 'min_polarity':min_polarity, 'max_polarity':max_polarity}

    def get_raw_text(self,corpora_path):
        path = corpora_path + self.get_path() + '.txt'
        with open(path, 'r', encoding="utf-8") as file:
            lines = file.readlines()
            raw = ''
            for line in lines:
                raw += line
        return raw

    def write_json(self):
        processed_dict = {'annotation_types':[], 'nodes': []}

        set = {annotation['Type'] for annotation in self.annotation_nodes}
        processed_dict['annotation_types'] = [anno for anno in set]

        for text_node in self.text_nodes:
            node_dict = {'id':text_node[0],'text':text_node[1],'annotations':[]}
            for annotation in self.annotation_nodes:
                if(int(text_node[0]) in range (int(annotation['StartNode']), int(annotation['EndNode'])+1)):
                    node_dict['annotations'].append(annotation)
            processed_dict['nodes'].append(node_dict)

        name = str(self.output +'/'+ self.docleaf + '.json')
        with open(name, 'w+') as outfile:
            json.dump(processed_dict,outfile)