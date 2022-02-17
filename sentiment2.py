import random
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Definition of First Plot
def sentimentplot1(ok):
    df_questions = pd.read_csv(Path(r'./Data/Processed/', 'SAplot1process.csv'))
    df_final = pd.DataFrame()
    # Get the languages chosen by the user
    for i in ok:
        temp = df_questions[df_questions["language"] == i]
        df_final = df_final.append(temp)

    x = df_final.language

    #Traces of Figures as selected by the user.
    trace1 = {
        'x': x,
        'y': df_final.pos,
        'name': 'Positive Sentiment',
        'type': 'bar'
    }
    trace2 = {
        'x': x,
        'y': df_final.neg,
        'name': 'Negative Sentiment',
        'type': 'bar'
    }
    data = [trace1, trace2]
    layout = {
        'xaxis': {'title': 'Languages'},
        'yaxis': {'title': 'Sentiment Score'},
        'barmode': 'relative',
        'title': 'How much positive and negative sentiments are there in questions from different programming languages'
    }
    #Figure is returned.
    fig = go.Figure(data=data, layout=layout)
    return fig

# Definition for the second plot
def sentimentplot2(ok):
    # Data Importing
    df_questions = pd.read_csv(Path(r'./Data/Processed/', 'SAplot2process.csv'))
    df_final2 = pd.DataFrame()
    # Getting only the relevant rows from the dataframe
    for i in ok:
        temp = df_questions[df_questions["language"] == i]
        df_final2 = df_final2.append(temp)


    # This snippet is for generating random colors based on the languages chosen
    color = ["#" +''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(ok))]
    df_final2['compound'] = df_final2['compound'].multiply(100)
    i = 0
    data = []
    # Getting Scatter Plot Traces depending on the number of languages that are chosen by the user.
    for o in ok:
        tempdf = df_final2[df_final2["language"]==o]
        trace1 = go.Scatter(
            y = tempdf.compound,
            x = tempdf.creation_date,
            mode = "lines+markers",
            name = o,
            marker = dict(color = color[i]),
            text= tempdf.language)

        data.append(trace1)
        i = i+1
    layout = dict(title = 'How positivity in questions has deviated over years',
                  xaxis= dict(title= 'Years',ticklen= 5,zeroline= False),
                  yaxis= dict(title= 'Polarity',ticklen= 5,zeroline= False)
                  )
    fig = go.Figure(data = data, layout = layout)
    return fig


if __name__ == "__main__":
    fig = sentimentplot2(['java', 'python'])
    fig.show()
