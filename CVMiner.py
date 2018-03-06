from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
import os
import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier as nbc
import nltk
import random

from sklearn.decomposition import PCA
from matplotlib import pyplot


# IMPORTING DATASET
accepted = [(x)+"\\"+(i) for x,y,z in os.walk(os.getcwd()) for i in z if (x[-3:] == 'pos' and i[-4:] == ".pdf")]
rejected = [(x)+"\\"+(i) for x,y,z in os.walk(os.getcwd()) for i in z if (x[-3:]  == 'neg' and i[-4:] == ".pdf")]

class CVparser:
    """ Extract text from PDF files for document classification
    sample = 'accepted' or 'rejected', load PDF files from corresponding folders
    status = 'pos' or 'neg', assign category to the entire batch """
    
    def __init__(self, sample, status):
        self.sample = sample
        self.status = status

    def getText(self):
        samples = [self.sample] if isinstance(self.sample, str) else self.sample
        results = []
        for file in samples:
            fp = open(file, 'rb')
            parser = PDFParser(fp)
            doc = PDFDocument()
            parser.set_document(doc)
            doc.set_parser(parser)
            doc.initialize('')
            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            # Process each page contained in the document.
            text = []
            for page in doc.get_pages():
                interpreter.process_page(page)
                layout = device.get_result()
                for lt_obj in layout:
                    if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                        text.append(str(lt_obj.get_text()))
            results.append([text,self.status])
        return results
                   
    def cleanText(self):        
        #REGEX EXPRESSIONS FOR ANONYMIZING DATA AND REMOVING NOISE
        html_parse = '(?:https?:\/\/)?(?:www\.)?[a-z0-9-]+\.(?:com|co|org)(?:\.[a-z]{2,3})?\/?'
        email_parse = '[a-z0-9-_.]+@[a-z0-9-]+\.[a-z]{2,3}(?:\.[a-z]{2,3})?'
        number_parse = '\+?[0-9]+'
        
        out = []
        for file in self.getText():
            text = file[0]
            text1 = [re.sub(email_parse, '', text[i]) for i in range(len(text))]
            text2 = [re.sub(html_parse, '', text1[i]) for i in range(len(text1))]
            text3 = [re.sub(number_parse, '', text2[i]) for i in range(len(text2))]
            text4 = [re.sub(r'\n', '', text3[i]) for i in range(len(text3))]
            out.append([text4, self.status])
        return out
    
    def getTokens(self):
        out = []
        for file in self.getText():
            text = file[0]
            text1 = [text[i].split() for i in range(len(text))]
            text2 = [i for i in text1 if i != []]
            text3 = [[i.lower() for i in j] for j in text2]
            out.extend(text3)
        return out
            
    def getTokens_bow(self):
        out = []
        stoplist = stopwords.words('english')
        for file in self.getText():
            text = file[0]
            text1 = [text[i].split() for i in range(len(text))]
            text2 = [i for j in text1 for i in j if isinstance(j, list)]
            text3 = [i for i in text2 if i != []]
            text4 = [i.lower() for i in text3]
            text5 = [i for i in text4 if i not in stoplist]
            out.extend([(text5, self.status)])
        return out
    
class learner:
    def __init__(self, sample):
        self.sample = sample
        
    def trainNBC(self):
        """ Baseline algorithm for benchmark purposes using CBOW """
        # SHUFFLE SAMPLE FOR RANDOM INITIALIZATION
        random.shuffle(self.sample)
        
        # DEFINE WORDS AS KEYS AND OCCURENCES AS VALUES
        word_features = FreqDist([x for y,z in self.sample for x in y])
        word_features = list(word_features.keys())#[:1000]

        # TERM-DOC MATRIX, SAMPLING TRAIN AND TEST SETS AT 80-20
        numtrain = int(len(self.sample) * 80 / 100)
        train_set = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in self.sample[:numtrain]]
        test_set = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in self.sample[numtrain:]]

        # TRAIN LEARNER AND RETURN PERFORMANCE MEASURES
        model = nbc.train(train_set)
        print(nltk.classify.accuracy(model, test_set)*100)
        print(model.show_most_informative_features(5))
        
    def trainW2V(self):
        # TRAIN LEARNER AND RETURN WORD VECTORS FOR THE VOCABULARY
        model = Word2Vec(self.sample, size = 100, window = 5, min_count = 1)
        return model[model.wv.vocab], list(model.wv.vocab)
        
        
out1 = CVparser(accepted, 'pos').getText()
out2 = CVparser(accepted, 'pos').cleanText()
out3 = CVparser(accepted, 'pos').getTokens()
out4 = CVparser(accepted, "pos").getTokens_bow()

# TRAIN MODEL, WORDS OCCURRING AT LEAST ONCE (min_count), VECTOR SIZE = 100, WINDOW SIZE = 5
wrdvector, words = learner(out3).trainW2V()

# GETTING WORD VECTORS AND CHECKING SIMILARITIES
#test["analytics"]
#test.most_similar("good", topn = 5)
#test.similarity(["good"],["analytics"])

# PLOT WORD VECTORS WITH PCA
zscores = PCA(n_components= 2).fit_transform(wrdvector)
pyplot.scatter(zscores[:, 0], zscores[:, 1])
for i,j in enumerate(words):
    pyplot.annotate(j, xy = (zscores[i, 0], zscores[i, 1]))
pyplot.show()

