import os
import re
from OPAM.Span import Span

class ArguingLexicon:
    data_path = os.path.expanduser('~/nltk_data/corpora')
    arguing_lexicon_path = '/arguing_lexicon/arglex_Somasundaran07/'
    test_path = 'patterntest.txt'
    macro_path_list = ['modals.tff', 'spoken.tff', 'wordclasses.tff', 'pronoun.tff', 'intensifiers.tff']
    lexicon_path_list = ['assessments.tff', 'authority.tff', 'causation.tff', 'conditionals.tff', 'contrast.tff', 'difficulty.tff', 'doubt.tff', 'emphasis.tff', 'generalization.tff', 'inconsistency.tff', 'inyourshoes.tff', 'necessity.tff', 'possibility.tff', 'priority.tff', 'rhetoricalquestion.tff', 'structure.tff', 'wants.tff']
    macro_pattern = '@(\w+)={(.*)}\n'
    class_pattern = '#class="(\w+)"'

    def __init__(self):
        self.macros ={}
        self.lexicon={}
        self.parse_macros()
        self.parse_lexicon()

    def parse_macros(self):
        for macro in ArguingLexicon.macro_path_list:
            temp_dict = {}
            curr = str(ArguingLexicon.data_path + ArguingLexicon.arguing_lexicon_path + macro)
            #print(curr)
            if(os.path.exists(curr)):
                with open(curr,'r') as macro_file:
                    classname = macro_file.readline()
                    classname = re.findall(ArguingLexicon.class_pattern,classname).pop()
                    lines = macro_file.readlines()
                    for line in lines:
                        if(re.match(ArguingLexicon.macro_pattern,line)):
                            try:
                                (key,val) = re.findall(ArguingLexicon.macro_pattern,line).pop()
                            except ValueError:
                                print('ValueError',line)
                            except:
                                print('\t\texcept')
                            finally:
                                val = val.replace(',','|')
                                val = val.replace(' ', '')
                                temp_dict[key.lower()] = val
                        self.macros[classname] = temp_dict
#print(str(macros))

    def getExpansion(self,macro_id):
        for key,val in self.macros.items():
            if(macro_id in val.keys()):
                return self.macros[key][macro_id]

    def parse_lexicon(self):
        for lex in ArguingLexicon.lexicon_path_list:
            curr = str(ArguingLexicon.data_path + ArguingLexicon.arguing_lexicon_path + lex)
            #print(curr)
            temp_pattern_list = []
            if(os.path.exists(curr)):
                with open(curr) as lex_file:
                    classname = lex_file.readline()
                    classname = re.findall(ArguingLexicon.class_pattern,classname).pop()
                    lines = lex_file.readlines()#[1:]
                    for line in lines:
                        line = line.replace('\n','')
                        line = line.lower()
                        if(line.__contains__('@')):
                            patt = '@(\w+)'
                            exp = re.findall(patt,line)
                            while(exp != [] ):
                                curr = exp.pop()
                                curr_expansion = self.getExpansion(curr)
                                line = line .replace(str('@'+curr), curr_expansion)
                        #line = re.compile(line)
                        temp_pattern_list.append(line)
                        ##print('adding...',classname)
                        self.lexicon[classname] = temp_pattern_list
#print(lexicon)

    def isAssessment(self,sentence):
        ret = self.findFragmentByType(sentence,'assessments')
        return ret

    def isAuthority(self,sentence):
        return self.findFragmentByType(sentence, 'authority')

    def isCausation(self,sentence):
        return self.findFragmentByType(sentence, 'causation')

    def isConditional(self,sentence):
        return self.findFragmentByType(sentence, 'conditionals')

    def isContrast(self,sentence):
        return self.findFragmentByType(sentence, 'contrast')

    def isDoubt(self,sentence):
        return self.findFragmentByType(sentence, 'doubt')

    def isEmphasis(self,sentence):
        return self.findFragmentByType(sentence, 'emphasis')

    def isGeneralization(self,sentence):
        return self.findFragmentByType(sentence, 'generalization')

    def isInyourshoes(self,sentence):
        return self.findFragmentByType(sentence, 'inyourshoes')

    def isInconsistency(self,sentence):
        return self.findFragmentByType(sentence,'inconsistency')

    def isNecessity(self,sentence):
        return self.findFragmentByType(sentence,'necessity')

    def isPossibility(self,sentence):
        return self.findFragmentByType(sentence, 'possibility')

    def isPriority(self,sentence):
        return self.findFragmentByType(sentence, 'priority')

    def isRhetoricalQuestion(self,sentence):
        return self.findFragmentByType(sentence, 'rhetoricalquestion')

    def isStructure(self,sentence):
        return self.findFragmentByType(sentence, 'structure')

    def isWants(self,sentence):
        return self.findFragmentByType(sentence, 'wants')

    def isDifficulty(self,sentence):
        return self.findFragmentByType(sentence, 'difficulty')

    def findFragmentByType(self,sentence,tipo):
        temp = []
        pattern_list = self.lexicon[tipo]
        for pattern in pattern_list:
            res = re.search(pattern, sentence)
            if (res != None):
                temp.append(res.group())
        return temp

    def func(self,oper):
        return re.findall('(is\w+)', oper).pop()

    # TODO: check if only right functions are called and find out why pattern-ish is captured
    def SentenceFragment(self,sentence):
        frag = [func for func in dir(ArguingLexicon) if callable(getattr(ArguingLexicon, func)) and not func.startswith("__")]
        ##found = []
        found={}
        for f in frag:
            if(re.match('is\w+',f)):
                method = self.func(f)
                #print('method:',method)
                try:
                    method = getattr(self, method)
                    #a.method(sentence)
                except AttributeError:
                    raise NotImplementedError(
                        "Class `{}` does not implement `{}`".format(self.__class__.__name__, method))
                try:
                    sentence = sentence.lower()
                    temp = method(sentence)
                    if(temp != []):
                        found[f]=temp
                except:
                    print('ERROR')
        return found

    def test(self):
            ##TODO:test file validation
            test = str(ArguingLexicon.data_path + ArguingLexicon.arguing_lexicon_path + ArguingLexicon.test_path)
            if(os.path.exists(test)):
                    with open(test,'r') as test_file:
                        lines = test_file.readlines()
                        for line in lines[0:28]: ## assessments
                                if(line.__contains__('#')):
                                    print('\n\t\t',line)
                                else:
                                    line = line.lower()
                                    res = self.isAssessment(line)
                                    print('testing::',line,' ##result:',res)
                        for line in lines[28:29]: ## authority
                            if(line.__contains__('#')):
                                print('\n\t\t',line)
                            else:
                                line = line.lower()
                                res = self.isAuthority(line)
                                print('testing::',line,' ##result:',res)
                        for line in lines[29:43]:  ## conditionals
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isConditional(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[43:52]:  ## contrast
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isContrast(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[52:58]: ## doubt
                            if(line.__contains__('#')):
                                print('\n\t\t',line)
                            else:
                                line = line.lower()
                                res = self.isDoubt(line)
                                print('testing::',line,' ##result:',res)
                        for line in lines[58:87]:  ## emphasis
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isEmphasis(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[87:93]: ## generalization
                            if(line.__contains__('#')):
                                print('\n\t\t',line)
                            else:
                                line = line.lower()
                                res = self.isGeneralization(line)
                                print('testing::',line,' ##result:',res)
                        for line in lines[93:98]:  ## inyourshoes
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isInyourshoes(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[98:116]:  ## inconsistency
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isInconsistency(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[116:128]:  ## necessity
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isNecessity(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[128:141]:  ## possibility
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isPossibility(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[141:149]:  ## priority
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isPriority(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[149:155]:  ## rhetoricalquestion
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isRhetoricalQuestion(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[155:158]:  ## structure
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isStructure(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[158:161]:  ## wants
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isWants(line)
                                print('testing::', line, ' ##result:', res)
                        for line in lines[161:179]:  ## difficulty
                            if (line.__contains__('#')):
                                print('\n\t\t', line)
                            else:
                                line = line.lower()
                                res = self.isDifficulty(line)
                                print('testing::', line, ' ##result:', res)


class ArguingSpan(Span):

    def __init__(self, arg, arguing_type,span):
        self.arg = arg
        self.arguing_type = arguing_type
        super(ArguingSpan, self).__init__(span.start, span.end)

    def __str__(self):
        return str(self.__class__.__name__ + ' arguing span: ' + self.arg +' type: '+ self.arguing_type + ' s: '+str(self.start) + ' e: ' + str(self.end))

test = False
if(test):
    a = ArguingLexicon()
    b = a.SentenceFragment('in order to')
    b = a.isConditional('hence it is always said that competition makes the society more effective.')
    b = a.isAssessment('Well, our understanding was that we could buy that here.')
    print(b)
    a.test()



