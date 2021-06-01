############################################################
# Train
############################################################    
def read_words(data, tokens_only=False):
    for i,doc in enumerate(data):
        if tokens_only:
            yield copy.deepcopy(doc)
        else:
            yield gensim.models.doc2vec.TaggedDocument(doc, [i])

d2v_data_words = data_words_final_t_neg
d2v_train_corpus = list(read_words(d2v_data_words))
d2v_reviews_df = df_appreviews_for_topics_neg

i=0   
d2v_train_corpus[i] # 80871 length
#d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, window =2, epochs=50, dm=0) # 3650/4000, test is pretty good
#d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, window =2, epochs=50, dm=1) # 3399/4000 , but test is very bad
#d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, window =5, epochs=50, dm=1) # 3024/4000 , but test is very bad
#d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, window =5, epochs=50, dm=0) # 3676/4000 , but test not so good
#d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=1, window =5, epochs=50, dm=0) # 3623/4000 , test is ok
d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=1, window =2, epochs=50, dm=0) #  
d2v_model.build_vocab(d2v_train_corpus)
d2v_model.train(d2v_train_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

match_train = []
for doc_id in range(len(d2v_train_corpus)):
    inferred_vector = d2v_model.infer_vector(d2v_train_corpus[doc_id].words)
    sims = d2v_model.docvecs.most_similar([inferred_vector], topn=5)
    top_doc_ids_all = [docid for docid, sim in sims]
    if doc_id in top_doc_ids_all:
        match_train.append(1)
    else:
        match_train.append(0)
print( sum(match_train))
print( len(match_train))

d2v_reviews_df.review_line
df_appreviews_all.columns.values

d2v_reviews_df.review_line
d2v_reviews_df.Date
d2v_reviews_df.Platform
d2v_reviews_df.Version
d2v_reviews_df.Review


input_line = []
match_line = []
match_sim = []
input_doc = []
match_doc = []
match_platform = []
match_version = []
match_date = []
doc_id=0
j=1
N_sim = 10



for doc_id in range(len(d2v_train_corpus)):
#for doc_id in range(20): 
    inferred_vector = d2v_model.infer_vector(d2v_train_corpus[doc_id].words)
    sims = d2v_model.docvecs.most_similar([inferred_vector], topn=N_sim)
    for j in range(N_sim):
        input_line.append(copy.deepcopy(   d2v_reviews_df.review_line[doc_id]      )) 
        match_line.append(copy.deepcopy(   d2v_reviews_df.review_line[sims[j][0]]  ))
        match_sim.append(sims[j][1])
        input_doc.append(d2v_reviews_df.Review[doc_id])
        match_doc.append(copy.deepcopy(d2v_reviews_df.Review[sims[j][0]]))
        match_platform.append(copy.deepcopy(d2v_reviews_df.Platform[sims[j][0]]))
        match_version.append(copy.deepcopy(d2v_reviews_df.Version[sims[j][0]]))
        match_date.append(copy.deepcopy(d2v_reviews_df.Date[sims[j][0]]))
    
d2v_sim_reviews_df= pd.DataFrame({'input_line':input_line, 'match_line':match_line, 'match_sim':match_sim, 'input_doc':input_doc, \
                                  'match_doc':match_doc, 'match_platform':match_platform, 'match_version':match_version, 'match_date':match_date })

d2v_sim_reviews_df.to_excel('d2v_similar_reviews_PayPal_low_rating.xlsx', index=False)


#############################################
# word2vec pre-trained
#############################################

# https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
glove_file = datapath(r'C:\Users\mkommaraju\Documents\Work\ML\Topic_modeling\pretrained\glove.6B\glove.6B.50d.txt')
tmp_file = get_tmpfile(r'C:\Users\mkommaraju\Documents\Work\ML\Topic_modeling\pretrained\glove.6B\glove.6B.50d_gensim.txt')

## one time conversion of formats
#from gensim.scripts.glove2word2vec import glove2word2vec
#glove2word2vec(glove_file, tmp_file)

p_w2v_model = KeyedVectors.load_word2vec_format(tmp_file)
data_words_nostops = data_words_nostops_t_neg
i=7
print(data_words_nostops[i])
sim_df=pd.DataFrame({'sim':np.ones(len(data_words_nostops))})
for j in range(len(data_words_nostops)):
    sim_df.sim[j] = p_w2v_model.wmdistance(data_words_nostops[i], data_words_nostops[j])
sim_df_sorted = sim_df.sort_values(by=['sim'])
for k in range(5):
    print(np.round(sim_df_sorted.sim[ sim_df_sorted.index.values[k] ],2))
    print(data_words_nostops[ sim_df_sorted.index.values[k] ])

# Above works but is very slow
    
##############################################
    
import dill
filename = 'step_4_competitor.pkl'
dill.dump_session(filename)
# and to load the session again:
#dill.load_session(filename)

from statsmodels.stats import
statsmodels.stats.proportion.proportion_confint(2734, 515)


