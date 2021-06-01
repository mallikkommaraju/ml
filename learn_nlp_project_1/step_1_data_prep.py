import pandas as pd
import numpy as np 
import os
from pprint import pprint
import matplotlib.pyplot as plt
import copy


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.chdir(r'C:\Users\mkommaraju\OneDrive - PayPal\MyDocuments_from_Laptop\Work\ML\NLP_Showcase')
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


df_train_validate = pd.read_csv('train.csv')
df_test           = pd.read_csv('test.csv')

df_train_validate = df_train_validate.loc[:1000]
df_test = df_test.loc[:1000]

df_train_validate.columns
df_train_validate['ABSTRACT'].head()

from sklearn.model_selection import train_test_split
df_train, df_validate = train_test_split(df_train_validate, test_size = 0.2)

df_train.reset_index(inplace=True)
df_validate.reset_index(inplace=True)
df_test.reset_index(inplace=True)

X_train = df_train['ABSTRACT']
Y_train = df_train[['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']]

X_validate = df_validate['ABSTRACT']
Y_validate = df_validate[['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']]

X_test = df_test['ABSTRACT']


############################################################
# Pre processing
############################################################    

import custom_preprocessing as cp
def my_custom_preprocessing(texts):
    texts= cp.my_simple_preprocessing(texts)
    texts = [w for w in cp.sent_to_words(texts)] 
    texts = cp.lemmatization(texts)
    texts = cp.remove_stopwords(texts)
    return texts

docs_train    = my_custom_preprocessing(X_train)
docs_validate = my_custom_preprocessing(X_validate)
docs_test     = my_custom_preprocessing(X_test)



import gensim.corpora as corpora   
import gensim
id2word = corpora.Dictionary(docs_train)
id2word.filter_extremes(no_below=10, keep_n=10000)
print('id2word length:' + str(len(id2word)))

# build the TFIDF model
tfidf_model = gensim.models.TfidfModel([id2word.doc2bow(text) for text in docs_train], id2word=id2word)

# build the corpus for train, validate and test data
corpus_train    = [id2word.doc2bow(text) for text in docs_train]
corpus_validate = [id2word.doc2bow(text) for text in docs_validate]
corpus_test     = [id2word.doc2bow(text) for text in docs_test]

# Build TF-IDF corpus for the train, validate and test data
corpus_tfidf_train    = tfidf_model[corpus_train]
corpus_tfidf_validate = tfidf_model[corpus_validate]
corpus_tfidf_test     = tfidf_model[corpus_test]

# build the X vectors for train, validate and test data
X_train_corpus_df       = pd.DataFrame(data = gensim.matutils.corpus2dense(corpus_train     , len(id2word.keys()) ).T, columns = list(id2word.values()))
X_validate_corpus_df    = pd.DataFrame(data = gensim.matutils.corpus2dense(corpus_validate  , len(id2word.keys()) ).T, columns = list(id2word.values()))
X_test_corpus_df        = pd.DataFrame(data = gensim.matutils.corpus2dense(corpus_test      , len(id2word.keys()) ).T, columns = list(id2word.values()))

# build the TFIDF X vectors for train, validate and test data
X_train_corpus_tfidf_df    = pd.DataFrame(data = gensim.matutils.corpus2dense(corpus_tfidf_train   , len(id2word.keys()) ).T, columns = list(id2word.values()))
X_validate_corpus_tfidf_df = pd.DataFrame(data = gensim.matutils.corpus2dense(corpus_tfidf_validate, len(id2word.keys()) ).T, columns = list(id2word.values()))
X_test_corpus_tfidf_df     = pd.DataFrame(data = gensim.matutils.corpus2dense(corpus_tfidf_test    , len(id2word.keys()) ).T, columns = list(id2word.values()))


#######################
# check data
#######################   

i=0
print(X_train[i])
print(docs_train[i])
[(id2word[id], freq) for id, freq in corpus_train[i]]
[(id2word[id], round(freq,2)) for id, freq in corpus_tfidf_train[i]]
Y_train.loc[i]
 
# columns/terms in the dictionary, top 20 words in the corpus
X_train_corpus_df.columns
X_train_corpus_df.mean(axis=0).sort_values(ascending =False).head(20)




#####################################
# save data so far
####################################

import dill
filename = 'step_1_data_prep.pkl'
dill.dump_session(filename)
# and to load the session again:
#dill.load_session(filename)



