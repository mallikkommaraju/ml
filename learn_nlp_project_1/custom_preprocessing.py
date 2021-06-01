import pandas as pd

#########################################################
# All Functions for pre-processing the text
#########################################################

import re
# convert emojis to text and replace any '_' with space
def my_simple_preprocessing(texts):
    
    texts_ = [re.sub('\S*@\S*\s?', '', str(sent) ) for sent in texts] # Remove Emails
    texts_ = [re.sub('\s+', ' ', str(sent)) for sent in texts_]       # Remove new line characters
    texts_ = [re.sub("\'", "", str(sent)) for sent in texts_]         # Remove distracting single quotes
    return texts_



import gensim.utils
from gensim.utils import simple_preprocess
# Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True,min_len=2, max_len=30))  # deacc=True removes punctuations




# https://spacy.io/usage/spacy-101
# https://github.com/explosion/spaCy/issues/4297
# https://github.com/explosion/spacy-models/releases//tag/en_core_web_sm-3.0.0
# https://github.com/explosion/spacy-models/releases//tag/en_core_web_lg-3.0.0
# Download tar.gz file and then run "pip install /local/path/to/en_core_web_sm-2.1.0.tar.gz"
import spacy
#import en_core_web_sm
#nlp = spacy.load('en_core_web_sm')
#nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
import en_core_web_lg
nlp = spacy.load('en_core_web_lg')

def lemmatization(texts, allowed_postags=['NOUN','VERB']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# remove stop words
from nltk.corpus import stopwords
#first time
#import nltk
#nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['journal', 'publish'])

def remove_stopwords(texts):
#    return [[word for word in simple_preprocess(str(doc), max_len=50) if word not in stop_words] for doc in texts]
    return [[word for word in doc if word not in stop_words] for doc in texts]

def getngrams(texts, n):
    ngram_list = []
    for text in texts:
        for s in text:
            if str(s).count('_') >= n-1:
                #print(s)
                ngram_list.append(s)
    return ngram_list

def main():
    text1 = u'Mark, whose email address is abc@gmail.com, asked for 10 apples grown in California qwertyupoialdafnnafgartbsfgytrdtddsrtsdfxc'
    text2 = u'I love apples grown in California'
    text3 = u'Steve wrote an article on physics'
    
    texts = [text1, text2, text3]
    
    texts=my_simple_preprocessing(texts)
     
    texts = [w for w in sent_to_words(texts)] #now, each text is an array of words
    print("\ntext after removing punctuation marks and extremely long words")
    print(texts)
    
    texts = lemmatization(texts)
    print("\ntext after lemmatization and pos filter")
    print(texts)
    
    texts = remove_stopwords(texts)
    print("\ntext after removing stop words")
    print(texts)
    
    print('\n\nTest POS & tags from Spacy')
    # https://spacy.io/usage/spacy-101#whats-spacy
    # https://spacy.io/api/annotation
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    doc3 = nlp(text3)
    
    text, lemma, pos_, tag_, dep_, shape_, is_alpha, is_stop, like_email, is_oov, has_vector, vector_norm= [], [], [], [], [], [], [], [], [],[],[],[]
    for token in doc1:
        text.append(token.text)
        pos_.append(token.pos_)
        tag_.append(token.tag_)
        dep_.append(token.dep_)
        shape_.append(token.shape_)
        is_alpha.append(token.is_alpha)
        is_stop.append(token.is_stop)
        like_email.append(token.like_email)
        is_oov.append(token.is_oov)
        has_vector.append(token.has_vector)
        vector_norm.append(token.vector_norm)
        lemma.append(token.lemma)
        
    df_doc1 = pd.DataFrame({'text':text,'pos_':pos_, 'tag_':tag_, 'dep_':dep_,'shape_':shape_,
                            'is_alpha':is_alpha, 'is_stop':is_stop,'like_email':like_email, 'is_oov':is_oov, 
                            'has_vector':has_vector, 'vector_norm':vector_norm,'lemma':lemma})
    
    print(df_doc1)
    
    print('\n\nTest similarity of docs from Spacy')
    print("similarity score: (doc1, doc2): {0}".format(doc1.similarity(doc2)))
    print("similarity score: (doc1, doc3): {0}".format(doc1.similarity(doc3)))
    print("similarity score: (doc2, doc3): {0}".format(doc2.similarity(doc3)))

if __name__ == "__main__":
    main()
