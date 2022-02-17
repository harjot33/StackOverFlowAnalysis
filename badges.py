import pandas as pd
from wordcloud import WordCloud
from pathlib import Path
import matplotlib.pyplot as plt
from users import *
import plotly.express as px

df_badges = pd.DataFrame()
df_user = pd.DataFrame()

def badges_init():
    global df_badges
    global df_user
    df_badges = pd.read_csv(Path(r'./Data/Processed/', 'badges.csv'))
    df_user = pd.read_csv(Path(r'./Data/Processed/', 'users.csv'))
     

def generate_badges_sunburst_chart():
    
    df=pd.DataFrame()
    df=df_badges.groupby(['name','class']).size().reset_index(name='count')
    df["class"].replace({1: "Gold", 2: "Silver", 3: "Bronze"}, inplace=True)
    
    fig =px.sunburst(
    df,path=['class', 'name'], values='count', title="Badges earned by user in different categories")
    
    return fig

    
def generate_location_chart(badge): 
    
    
    df_users= df_user.rename(columns={'id': 'user_id'})
    result = pd.merge(df_users, df_badges, on='user_id')
    badge_1=result[result['class']==1]
    badge_2=result[result['class']==2]
    badge_3=result[result['class']==3]
    
    if(badge=='Gold'):
        new_df = badge_1.groupby('location').size().reset_index(name='count')
    elif(badge=="Silver"):
        new_df = badge_2.groupby('location').size().reset_index(name='count')
    elif(badge=="Bronze"):
        new_df = badge_3.groupby('location').size().reset_index(name='count')
        
    new_df["country_code"] = get_country_code(new_df, pd.unique(new_df['location']))
    fig = go.Figure(data=go.Choropleth(
         locations=new_df['country_code'],
         z=new_df['count'],
         hovertext=new_df['location'],
         colorscale="Blues",
         autocolorscale=False,
         reversescale=True,
         marker_line_color='white',
         marker_line_width=0.5,
         colorbar_title='User Count'
     ))
    fig.update_layout(
         title_text='Stack Overflow Users',
         geo=dict(
             showframe=False,
             showcoastlines=False,
             projection_type='equirectangular'
         ),
         clickmode='event+select'
     )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig


def generate_bar_chart_badges():
    
    df_users= df_user.rename(columns={'id': 'user_id'})
    result = pd.merge(df_users, df_badges, on='user_id')
    result["class"].replace({1: "Gold", 2: "Silver", 3: "Bronze"}, inplace=True)
    result=result.groupby(['location', 'class'])['class'].count().reset_index(name='count')
    
    fig = px.bar(result, x="location", y="count", color="class", title="badges distribution based on location")
    return fig
   
    
   
    
    
