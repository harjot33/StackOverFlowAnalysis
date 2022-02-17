import numpy as np
import pandas as pd
import pycountry
from data_cleaning import *
from data_profile import *
import os.path

def process():

    print("Checking Data")

    if os.path.isfile("Data/Processed/comments.csv") == False:
        print("Processing...")
        preprocessing_comments()
    if os.path.isfile("Data/Processed/post_answers.csv") == False:
        print("Processing...")
        proprocessing_posts_answers()
    if os.path.isfile("Data/Processed/post_questions.csv") == False:
        print("Processing...")
        processing_posts_questions()
    if os.path.isfile("Data/Processed/badges.csv") == False:
        print("Processing...")
        processing_badges()
    if os.path.isfile("Data/Processed/tags.csv") == False:
        print("Processing...")
        processing_tags()
    if os.path.isfile("Data/Processed/users.csv") == False:
        print("Processing...")
        processing_users()
    print("Preprocessing Completed")


def preprocessing_comments():

    df_comments = pd.read_csv("Data/Raw/comments.csv")

    cols_to_select = ['id', 'text', 'creation_date', 'post_id', 'user_id', 'score']

    df_comments = df_comments[cols_to_select]

    df_comments = df_comments.dropna()

    df_comments['creation_date'] = pd.to_datetime(df_comments['creation_date'], format='%Y/%m/%d')

    df_comments.to_csv("Data/Processed/comments.csv", index=False)

def proprocessing_posts_answers():
    df_ans = pd.read_csv("Data/Raw/post_answers.csv")

    cols_to_keep = ['id',
                    'body',
                    'comment_count',
                    'creation_date',
                    'last_activity_date',
                    'last_edit_date',
                    'last_editor_user_id',
                    'parent_id',
                    'score']

    df_ans = df_ans[cols_to_keep]

    df_ans = df_ans.dropna()

    df_ans['creation_date'] = pd.to_datetime(df_ans['creation_date'], format='%Y/%m/%d')
    df_ans['last_activity_date'] = pd.to_datetime(df_ans['last_activity_date'], format='%Y/%m/%d')
    df_ans['last_edit_date'] = pd.to_datetime(df_ans['last_edit_date'], format='%Y/%m/%d')

    df_ans['body'] = df_ans['body'].apply(lambda x: x.strip("<p></p>"))

    df_ans.to_csv("Data/Processed/post_answers.csv", index=False)

def processing_posts_questions():


    df_ques = pd.read_csv("Data/Raw/post_questions.csv")

    cols_selected = ['id',
                     'title',
                     'body',
                     'answer_count',
                     'comment_count',
                     'creation_date',
                     'last_activity_date',
                     'last_edit_date',
                     'last_editor_user_id',
                     'owner_user_id',
                     'post_type_id',
                     'score',
                     'tags',
                     'view_count']

    df_ques = df_ques[cols_selected]

    df_ques = df_ques.dropna()

    df_ques['creation_date'] = pd.to_datetime(df_ques['creation_date'], format='%Y/%m/%d')
    df_ques['last_activity_date'] = pd.to_datetime(df_ques['last_activity_date'], format='%Y/%m/%d')
    df_ques['last_edit_date'] = pd.to_datetime(df_ques['last_edit_date'], format='%Y/%m/%d')

    df_ques['body'] = df_ques['body'].apply(lambda x: x.strip("<p></p>"))

    df_ques.to_csv("Data/Processed/post_questions.csv", index=False)

def processing_badges():

    df_badges = pd.read_csv('Data/Raw/badges.csv')

    cols = get_numeric_columns(df_badges)
    for col in cols:
        df_badges = fix_numeric_wrong_values(df_badges, col, WrongValueNumericRule.MUST_BE_POSITIVE)

    df_badges = df_badges.dropna()

    for col in cols:
        df_badges[col] = df_badges[col].astype(np.int64)

    df_badges['date'] = pd.to_datetime(df_badges['date'], format='%Y/%m/%d')

    df_badges.to_csv("Data/Processed/badges.csv", index=False)

def processing_tags():

    df_tags = pd.read_csv("Data/Raw/tags.csv")

    cols = get_numeric_columns(df_tags)
    for col in cols:
        df_tags = fix_numeric_wrong_values(df_tags, col, WrongValueNumericRule.MUST_BE_POSITIVE)

    df_tags = df_tags.dropna()

    for col in cols:
        df_tags[col] = df_tags[col].astype(np.int64)

    df_tags.to_csv("Data/processed/tags.csv", index=False)

def processing_users():

    print("Processing of user started!!")

    df_users_big = pd.read_csv("Data/Raw/user.csv")

    df_users_big = df_users_big.drop(['about_me', 'age', 'profile_image_url', 'website_url'], axis=1)

    cols = get_numeric_columns(df_users_big)
    for col in cols:
        df_users_big = fix_numeric_wrong_values(df_users_big, col, WrongValueNumericRule.MUST_BE_POSITIVE)

    df_users_big = df_users_big.dropna()

    cols = get_numeric_columns(df_users_big)

    for col in cols:
        df_users_big[col] = df_users_big[col].astype(np.int64)

    df_users_big['creation_date'] = pd.to_datetime(df_users_big['creation_date'], format='%Y/%m/%d')
    df_users_big['last_access_date'] = pd.to_datetime(df_users_big['last_access_date'], format='%Y/%m/%d')

    print("okay Check")

    mask = df_users_big['location'].str.contains("[A-za-z\s,]")
    df_users_big[mask]['location']

    countries = []
    for country in list(pycountry.countries):
        countries.append(country.name)

    countries.sort()

    countries.append('UK')
    countries.append('USA')

    for i in df_users_big.index:
        for country in countries:
            loc = str(df_users_big['location'][i])
            if country in loc:
                if country is 'USA':
                    df_users_big.at[i, 'location'] = 'United States'
                elif country is 'UK':
                    df_users_big.at[i, 'location'] = 'United Kingdom'
                else:
                    df_users_big.at[i, 'location'] = country

    indieces = []
    for i in df_users_big.index:
        flag = False
        loc = str(df_users_big['location'][i])
        for country in countries:
            if country == loc:
                flag = True
        if flag == False:
            indieces.append(i)

    df_users_big = df_users_big.drop(indieces)

    df_users_big.to_csv("Data/processed/users.csv",index=False)




