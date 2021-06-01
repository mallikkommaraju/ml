#################################
# Topic quality - coherence
#################################

corpus = corpus_tfidf_train
docs = docs_train
id2word = id2word

# https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

start=2; limit=10;  step=1;
models_per_level = 5
c_v    = []
model_list = []
random_seed_list = []
df_c_v = pd.DataFrame()
model_list_i = 0
model_list_index = []

#num_topics=4
for num_topics in range(start, limit, step):
    c = -float("inf")
    i_max = 0
    c_list = []
    i=0
    for i in range(models_per_level):
        print('n:'+str(num_topics)+' i:'+str(i))
        model_i = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=num_topics,alpha='auto',per_word_topics=False, minimum_probability=0, random_state=i)
        c_i = CoherenceModel(model=model_i, texts=docs, dictionary=id2word, coherence='c_v' , processes =1).get_coherence()
        c_list.append(c_i)
        if c_i>c:
            model = copy.deepcopy(model_i)
            c = c_i
            i_max = i
    c_v.append (c)
    model_list.append(model)
    model_list_index.append(model_list_i)
    model_list_i += 1
    random_seed_list.append(i_max)
    df_c_v["n_"+str(num_topics)] = c_list


plt.figure(figsize=(15,5))
df_c_v.boxplot()
plt.xlabel("Num Topics")
plt.ylabel("Coherence score c_v")
plt.show()

plt.figure(figsize=(15,5))
plt.plot(range(start, limit, step), c_v,'r')
plt.xlabel("Num Topics")
plt.ylabel("Coherence score c_v")
plt.show()

for i in range(0, len(model_list), 1):
    print("i:{0}, cv:{1}".format(i,c_v[i]))

# chose number of categories based on lowest coherence score
len(model_list)
i=3
c_v[i]
model = model_list[i]


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
#pyLDAvis.enable_notebook()
vis = gensimvis.prepare(model, corpus, id2word, sort_topics=False, mds='tsne')
#pyLDAvis.show(vis)
pyLDAvis.save_html(vis, 'lda_vis_'+str(model.num_topics)+'_topics.html')

vis0 = vis[0] # topic	x	y	topics	cluster	Freq
vis1 = vis[1] # term	Category	Freq	Term	Total	loglift	logprob
vis2 = vis[2] # Topic, Freq, Term



#https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud.fit_words
from wordcloud import WordCloud
w = WordCloud(background_color='white', relative_scaling=1, colormap ='tab10')
w = WordCloud(background_color='white', relative_scaling=1, color_func=lambda *args, **kwargs: (0,0,255))
for t in range(model.num_topics):
    plt.figure(figsize=(4,4))
    plt.imshow(w.fit_words(dict(model.show_topic(t, 5)) ))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()
# changed here for sure




model.print_topics()
from itertools import compress
topic_key_words = []
max_key_works = 7
for t in range(model.num_topics):
    ss = [s for s, v in model.show_topic(t, max_key_works) if s!='pron']
    vv = [v for s, v in model.show_topic(t, max_key_works) if s!='pron']
    cum_percentile = np.cumsum(vv)/sum(vv)
    key_words = ss[0]
    for i in list(range(1,len(ss),1)):
        if cum_percentile[i]<0.8:
            key_words = key_words + ' ' + ss[i]
#    topic_key_words.append(' '.join([s for s, v in model.show_topic(t, 5) if s!='pron'] ))
    topic_key_words.append(key_words)
pprint(topic_key_words)


corpus_topics_df = pd.DataFrame(columns = topic_key_words)
for i, doc_topics in enumerate(list(model[corpus])):
    corpus_topics_df.loc[i] = np.round([p for t,p in doc_topics],2)

corpus_dominant_topic_df = pd.DataFrame({'Dominant Topic':corpus_topics_df.idxmax(axis=1), \
                                         'Dominant Topic Percent':corpus_topics_df.max(axis=1)})

from sklearn.manifold import TSNE
topic_tsne_2d = TSNE(n_components=2, random_state=0).fit_transform(corpus_topics_df)
topic_tsne_2d_df = pd.DataFrame(data = topic_tsne_2d, columns = ['tsne_x','tsne_y'])
plt.scatter(topic_tsne_2d[:,0], topic_tsne_2d[:,1], alpha=0.1)
plt.show()

# https://stackoverflow.com/questions/57242208/how-to-resolve-the-error-module-umap-has-no-attribute-umap-i-tried-installi
#import umap
import umap.umap_ as umap
import matplotlib.cm as cm
umap_n = max(round(corpus_topics_df.shape[0]*0.01),5)
topic_umap_2d = umap.UMAP(n_neighbors=umap_n, min_dist=0.3, n_components=2).fit_transform(corpus_topics_df.values)
topic_umap_2d_df = pd.DataFrame(data = topic_umap_2d, columns = ['umap_x','umap_y'])
plt.scatter(topic_umap_2d[:,0], topic_umap_2d[:,1], alpha=0.1)
plt.show()





import dill
filename = 'step_3_topic_modeling.pkl'
dill.dump_session(filename)
# and to load the session again:
#dill.load_session(filename)

