from gensim.models import Word2Vec
from pathlib import Path
from nltk.cluster import KMeansClusterer
import nltk
import pandas as pd
from string import digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from matplotlib import pyplot
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
from nltk import FreqDist
from nltk.corpus import stopwords
stoplist = stopwords.words('english')

def word2vec():
    df= pd.read_csv(Path('./Data/Processed/', 'post_questions.csv'))
    text=df['body']
    
    corpus=[]
    remove_list=["</p>", "<p>", "<code>", "</code>", "<ol>", "<li>",";", "&", "<a", "=", "/", "<pre>", "a", "the", "I", "have", "has", "@", "<ul>", "$",'.', "is", "are", "in", "tht","on"]
    remove_list= remove_list + stoplist
    
    
    for post in text:
        sentences=str(post).split("</p>") 
        for i in sentences:
            for char in remove_list:  
                remove_digits = str.maketrans('', '', digits)
                i = i.translate(remove_digits)
                i=i.replace(char, "")
                
            
            corpus.append(i.split(" "))
            
            
    model = Word2Vec(corpus, window=10, min_count=10)
    
    vocab=list(model.wv.index_to_key)
    X=model.wv[vocab]
    NUM_CLUSTERS=5
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    
    
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    cluster_df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    cluster_df['word']=vocab
    cluster_df['label']=assigned_clusters
    cluster_df=cluster_df[110:2000]
    # Generating vector.csv file and storing just to reduce the model training time
    cluster_df.to_csv('vector.csv')


def vector_fig():
    df=pd.read_csv(Path('./Data/Processed/', 'vector.csv'))
    fig = px.scatter(df, x="x", y="y", color="label" , text='word')

    return fig

    
    



