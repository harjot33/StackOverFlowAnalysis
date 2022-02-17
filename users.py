import pandas as pd
from pathlib import Path
import pycountry
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import datetime
import collections

df = pd.DataFrame()

"""
Data is loaded at once in the dataframe which is used throughout the program
"""


def users_init():
    global df
    df = pd.read_csv(Path('./Data/Processed/', 'users.csv'))

"""
Country names are translated into country codes
"""


def get_country_code(countries_list):
    countries = {}
    for country in pycountry.countries:
        countries[country.name] = country.alpha_3
    country_code = [countries.get(country, 'Unknown code') for country in countries_list]
    return country_code

"""
Users concentration based on location is plotted
"""


def load_user_by_country():
    new_df = df.groupby('location').size().reset_index(name='count')
    new_df["country_code"] = get_country_code(pd.unique(new_df['location']))
    fig = go.Figure(data=go.Choropleth(
        locations=new_df['country_code'],
        z=new_df['count'],
        hovertext=new_df['location'],
        colorscale="greens",
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

"""
Active users per year are calculated for a particular location
"""


def active_users_per_year(location):
    global df
    df["country_code"] = get_country_code(df['location'])
    location_df = df[df["country_code"] == location].copy()
    original_year = []
    last_year = []
    for val in location_df["creation_date"].values:
        try:
            original_year.append(datetime.datetime.strptime(val, '%Y-%m-%d %H:%M:%S.%f+00:00').year)
        except ValueError:
            new_val = val.split("+")
            val = new_val[0]+".0+"+new_val[1]
            original_year.append(datetime.datetime.strptime(val, '%Y-%m-%d %H:%M:%S.%f+00:00').year)
    for val in location_df["last_access_date"].values:
        try:
            last_year.append(datetime.datetime.strptime(val, '%Y-%m-%d %H:%M:%S.%f+00:00').year)
        except ValueError:
            new_val = val.split("+")
            val = new_val[0] + ".0+" + new_val[1]
            last_year.append(datetime.datetime.strptime(val, '%Y-%m-%d %H:%M:%S.%f+00:00').year)
    location_df["created_year"] = original_year
    location_df["last_access_year"] = last_year
    location_df["years_active"] = location_df["last_access_year"] - location_df["created_year"]
    active_users = {}
    years = pd.unique(location_df[['created_year', 'last_access_year']].values.ravel('K'))
    for val in years:
        new_df = location_df[(location_df["created_year"] <= val) & (location_df["last_access_year"] >= val)]
        count = len(new_df.index)
        active_users[val] = count
    active_users = collections.OrderedDict(sorted(active_users.items()))
    active_users_years = list(active_users.keys())
    active_users_count = list(active_users.values())
    fig = go.Figure(data=go.Scatter(x=active_users_years, y=active_users_count))
    fig.update_layout(
        title="Number of active users per year in " + location_df["location"].iloc[0],
        xaxis_title="Years",
        yaxis_title="Number of users"
    )
    return fig


def generate_user_upvotes():
    
    country = df.location.unique()
    country_name=list()
    view_country=list()
    votes=list()
    
    for c in country:
        
        temp_df=df.loc[df['location'] == c]
        country_name.append(c)
        up_vote=temp_df.groupby('location')['up_votes'].sum()
        votes.append(up_vote[0])
        views=temp_df.groupby('location')['views'].sum()
        view_country.append(views[0])
       
    country_code = get_country_code(country_name)
    fig = go.Figure(data=go.Choropleth(
        locations=country_code,
        z=votes,
        hovertext=country_name,
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


def generate_tree_map_reputation():

    df_rep_sorted = df.sort_values(by=['reputation'], ascending=False)

    df_rep_sorted = df_rep_sorted.head(1000)

    fig = px.treemap(df, path=[df_rep_sorted['location'], df_rep_sorted['display_name'],
                               df_rep_sorted['up_votes']], values=df_rep_sorted['up_votes'],
                     color=df_rep_sorted['up_votes'],
                     color_continuous_scale='RdBu',
                     color_continuous_midpoint=np.average(df_rep_sorted['up_votes']),
                     title="Top 1000 users based on Reputation")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

    return fig

def generate_tree_map_upvotes():

    df_votes_sorted = df.sort_values(by=['up_votes'], ascending=False)

    df_votes_sorted = df_votes_sorted.head(1000)

    fig = px.treemap(df, path=[df_votes_sorted['location'], df_votes_sorted['display_name'],
                               df_votes_sorted['up_votes']], values=df_votes_sorted['up_votes'],
                     color=df_votes_sorted['up_votes'],
                     color_continuous_scale='RdBu',
                     color_continuous_midpoint=np.average(df_votes_sorted['up_votes']),
                     title="Top 1000 users based on up-votes")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

    return fig


def generate_user_reputation():
    
    df_user_reputation=df.groupby(['location'])['reputation'].sum().reset_index(name='reputation')
    df_user_reputation=df_user_reputation.sort_values('reputation')
   
    fig = px.funnel(df_user_reputation[-10:], x='reputation', y='location')
    
    
    return fig
