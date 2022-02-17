import random
from pathlib import Path

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.widgets import Button, Slider
from plotly.offline import iplot

# Get the questions dataset and then select rows depending on the needs of the plot
def sentimentplot1data():
    df_questions = pd.read_csv(Path(r'C:\Users\Satnam\PycharmProjects\csci6612-va-cosmos', 'questions.csv'))
    langarray = df_questions["tags"].to_numpy()

    langs = []
    langdict = {}
    for l in langarray:
        langs.append(l.split("|")[0])

    df_questions["language"] = langs
    ok = df_questions["language"].value_counts()
    ok = ok.iloc[:20]
    comparray = []
    negarray = []
    neuarray = []
    posarray = []
    df_final = pd.DataFrame()
    df_final2 = pd.DataFrame()
    dfio = []
    yearly = []
    for o in ok.index.to_list():
        chosendf = df_questions[df_questions["language"] == o]
        chosendf['creation_date'] = pd.to_datetime(chosendf['creation_date'])
        # Creating resamples from the dataframe - Getting the Compound Score, Positive Score, Negative, Neutral Score and the User Score
        jj = chosendf.resample('Y', on='creation_date').compound.mean()
        jneg = chosendf.resample('Y', on='creation_date').neg.mean()
        jpos = chosendf.resample('Y', on='creation_date').pos.mean()
        jneu = chosendf.resample('Y', on='creation_date').neu.mean()
        jscores = chosendf.resample('Y', on='creation_date').score.mean()
        jj = jj.reset_index()
        jneg = jneg.reset_index()
        jpos = jpos.reset_index()
        jneu = jneu.reset_index()
        meancompound = jj["compound"].mean() * 100
        comparray.append(meancompound)
        meanneg = jneg["neg"].mean() * 100
        negarray.append(meanneg)
        meanneu = jneu["neu"].mean() * 100
        neuarray.append(meanneu)
        meanpos = jpos["pos"].mean() * 100
        posarray.append(meanpos)
        jj["language"] = o
        jj["pos"] = jpos["pos"]
        jj["neg"] = jneg["neg"]
        df_final2 = df_final2.append(jj,ignore_index=True)
        dfo = [o,meanpos,meanneg]
        dfio.append(dfo)


    df_final=pd.DataFrame(dfio,columns=['language','pos','neg'])
    df_final.to_csv('SAplot1process.csv')

# Get the questions dataset and then select rows depending on the needs of the plot
def sentimentplot2data():
    df_questions = pd.read_csv(Path(r'C:\Users\Satnam\PycharmProjects\csci6612-va-cosmos', 'questions.csv'))
    langarray = df_questions["tags"].to_numpy()

    df_final2 = pd.DataFrame()
    langs = []
    langdict = {}

    for l in langarray:
        langs.append(l.split("|")[0])
    df_questions["language"] = langs
    ok = df_questions["language"].value_counts()
    ok = ok.iloc[:20]

    df_questions["language"] = langs
    for o in ok.index.to_list():
        chosendf = df_questions[df_questions["language"] == o]
        chosendf['creation_date'] = pd.to_datetime(chosendf['creation_date'])
        # Creating resamples from the dataframe - Getting the Compound Score, Positive Score, Negative, Neutral Score and the User Score
        jj = chosendf.resample('Y', on='creation_date').compound.mean()
        jneg = chosendf.resample('Y', on='creation_date').neg.mean()
        jpos = chosendf.resample('Y', on='creation_date').pos.mean()
        jneu = chosendf.resample('Y', on='creation_date').neu.mean()
        jscores = chosendf.resample('Y', on='creation_date').score.mean()
        jscores = jscores.reset_index()
        jj = jj.reset_index()
        jneg = jneg.reset_index()
        jpos = jpos.reset_index()
        jj["language"] = o
        jj["pos"] = jpos["pos"]
        jj["neg"] = jneg["neg"]
        jj["score"] = jscores["score"]
        df_final2 = df_final2.append(jj,ignore_index=True)

    df_final2.to_csv('SAplot2process.csv')

# Get the answers dataset and then select rows depending on the needs of the plot
def sentimentplot3data():
    df_questions = pd.read_csv(Path(r'C:\Users\Satnam\PycharmProjects\csci6612-va-cosmos', 'answers.csv'))
    df_questions['creation_date'] = pd.to_datetime(df_questions['creation_date'])
    chosendf = df_questions
    chosendf['creation_date'] = pd.to_datetime(chosendf['creation_date'])
    # Creating resamples from the dataframe - Getting the Compound Score, Positive Score, Negative, Neutral Score and the User Score
    jj = chosendf.resample('M', on='creation_date').compound.mean()
    jneg = chosendf.resample('M', on='creation_date').neg.mean()
    jpos = chosendf.resample('M', on='creation_date').pos.mean()
    jneu = chosendf.resample('M', on='creation_date').neu.mean()
    jscores = chosendf.resample('M', on='creation_date').score.mean()
    jj = jj.reset_index()
    jpos = jpos.reset_index()
    jscores = jscores.reset_index()
    jneg = jneg.reset_index()
    jj["pos"] = jpos["pos"]
    jj["neg"] = jneg["neg"]
    jj["score"] = jscores["score"]
    jj.to_csv('SAplot3process.csv')

# Get the comments dataset and then select rows depending on the needs of the plot
def sentimentplot6data():
    df_questions = pd.read_csv(Path(r'C:\Users\Satnam\PycharmProjects\csci6612-va-cosmos', 'comments.csv'))
    df_questions['creation_date'] = pd.to_datetime(df_questions['creation_date'])
    chosendf = df_questions
    chosendf['creation_date'] = pd.to_datetime(chosendf['creation_date'])
    # Creating resamples from the dataframe - Getting the Compound Score, Positive Score, Negative, Neutral Score and the User Score
    jj = chosendf.resample('M', on='creation_date').compound.mean()
    jneg = chosendf.resample('M', on='creation_date').neg.mean()
    jpos = chosendf.resample('M', on='creation_date').pos.mean()
    jneu = chosendf.resample('M', on='creation_date').neu.mean()
    jscores = chosendf.resample('M', on='creation_date').score.mean()
    jj = jj.reset_index()
    jpos = jpos.reset_index()
    jscores = jscores.reset_index()
    jneg = jneg.reset_index()
    jj["pos"] = jpos["pos"]
    jj["neg"] = jneg["neg"]
    jj["score"] = jscores["score"]
    jj.to_csv('SAplot6process.csv')


if __name__ == "__main__":
    sentimentplot6data()