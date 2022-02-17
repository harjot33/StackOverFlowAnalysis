import random
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Definition of third plot
def sentimentplot3(year):
    # Data Import
    df_questions = pd.read_csv(Path(r'./Data/Processed/', 'SAplot3process.csv'))
    df_questions['creation_date'] = pd.to_datetime(df_questions['creation_date'])
    # Get the relevant data
    jj = df_questions[df_questions["creation_date"].dt.year == year]

    # For getting random colors for the graph traces
    color = ["#" +''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(4)]
    i = 0
    data = []
    jj['compound'] = jj['compound'].multiply(100)
    trace1 = go.Scatter(
        y = jj.compound,
        x = jj.creation_date,
        mode = "lines+markers",
        name = "Answer Polarity",
        marker = dict(color = color[i]),
        )

    data.append(trace1)
    i = i+1
    layout = dict(title = 'How are the answers changing over time in terms of sentiments?',
                  xaxis= dict(title= 'Duration',ticklen= 5,zeroline= False),
                  yaxis= dict(title= 'Answer Polarity',ticklen= 5,zeroline= False)
                  )
    fig = go.Figure(data = data, layout = layout)

    return fig

# Definition for fourth plot
def sentimentplot4(year):
    # The questions of the user are firstly retrieved
    df_questions = pd.read_csv(Path(r'./Data/Processed/', 'SAplot3process.csv'))
    df_questions['creation_date'] = pd.to_datetime(df_questions['creation_date'])
    # Get the relevant data from the dataframe
    jj = df_questions[df_questions["creation_date"].dt.year == year]

    color = ["#" +''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(4)]
    i = 0
    data = []
    jj['compound'] = jj['compound'].multiply(100)
    jj['pos'] = jj['pos'].multiply(100)
    jj['neg'] = jj['neg'].multiply(100)

    # Define Traces
    trace0 = go.Box(
        y=jj.pos,
        name = 'Positive Sentiment Answers',
        marker = dict(
            color = 'rgb(12, 12, 140)',
        )
    )
    trace1 = go.Box(
        y=jj.neg,
        name = 'Negative Sentiment Answers',
        marker = dict(
            color = 'rgb(12, 128, 128)',
        )
    )
    layout = dict(title = 'Annual Comparison of Positive and Negative Sentiments in answers',
                  xaxis= dict(title= 'User Answers',ticklen= 5,zeroline= False),
                  yaxis= dict(title= 'Polarity Score',ticklen= 5,zeroline= False)
                  )
    data = [trace0, trace1]
    fig = go.Figure(data = data, layout=layout)

    # Fig is returned
    return fig


# This is the definition for the fifth plot
def sentimentplot5(year1,year2):
    df_questions = pd.read_csv(Path(r'./Data/Processed/', 'SAplot3process.csv'))
    # Get the data and then select the required rows depending on what the user chose
    df_questions['creation_date'] = pd.to_datetime(df_questions['creation_date'])
    jj = df_questions[df_questions["creation_date"].dt.year == year1]
    jj2 = df_questions[df_questions["creation_date"].dt.year == year2]


    # Define Traces
    trace1 = go.Histogram(
        x=jj.score,
        opacity=0.75,
        name = year1,
        marker=dict(color='rgba(171, 50, 96, 0.6)'))
    trace2 = go.Histogram(
        x=jj2.score,
        opacity=0.75,
        name = year2,
        marker=dict(color='rgba(12, 50, 196, 0.6)'))

    data = [trace1, trace2]
    # Define Layout
    layout = go.Layout(barmode='overlay',
                       title='Comparison of User Score for year '+str(year1)+' and '+str(year2),
                       xaxis=dict(title='Answer Score'),
                       yaxis=dict( title='Count'),
                       )
    fig = go.Figure(data=data, layout=layout)

    return fig

# This is the definition for the 6th plot
def sentimentplot6(year1):
    # Get the data and then select the required rows depending on what the user chose
    df_questions = pd.read_csv(Path(r'./Data/Processed/', 'SAplot6process.csv'))
    df_questions['creation_date'] = pd.to_datetime(df_questions['creation_date'])
    commentframe = df_questions[df_questions["creation_date"].dt.year == year1]
    commentframe['compound'] = commentframe['compound'].multiply(100)
    commentframe['pos'] = commentframe['pos'].multiply(100)
    commentframe['neg'] = commentframe['neg'].multiply(100)
    commentframe['score'] = commentframe['score'].multiply(100)


    # Define Traces
    trace1 = go.Scatter(
        x=commentframe.creation_date,
        y=commentframe.pos,
        name = "Positive Comments"
    )
    trace2 = go.Scatter(
        x=commentframe.creation_date,
        y=commentframe.neg,
        xaxis='x2',
        yaxis='y2',
        name = "Negative Comments"
    )
    trace3 = go.Scatter(
        x=commentframe.creation_date,
        y=commentframe.compound,
        xaxis='x3',
        yaxis='y3',
        name = "Compound Score"
    )
    trace4 = go.Scatter(
        x=commentframe.creation_date,
        y=commentframe.score,
        xaxis='x4',
        yaxis='y4',
        name = "User Comments"
    )
    data = [trace1, trace2, trace3, trace4]
    # Define Layout
    layout = go.Layout(
        xaxis=dict(title='Duration',
            domain=[0, 0.45]
        ),
        yaxis=dict(
            title='Positivity Score',
            domain=[0, 0.45]
        ),
        xaxis2=dict(title='Duration',
            domain=[0.55, 1]
        ),
        xaxis3=dict(title='Duration',
            domain=[0, 0.45],
            anchor='y3'
        ),
        xaxis4=dict(title='Duration',
            domain=[0.55, 1],
            anchor='y4'
        ),
        yaxis2=dict(
            title='Negativity Score',
            domain=[0, 0.45],
            anchor='x2'
        ),
        yaxis3=dict(
            title='Total Compound Score',
            domain=[0.55, 1]
        ),
        yaxis4=dict(
            title='User Comment Score',
            domain=[0.55, 1],
            anchor='x4'
        ),
        title = 'Positive, Negative, Total Polarity and User Score Comparison Annually'
    )
    fig = go.Figure(data=data, layout=layout)

    return fig
if __name__ == "__main__":
    fig = sentimentplot6(2012)
    fig.show()

