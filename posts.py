from pathlib import Path
import pandas as pd
import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

post_questions_encoded_tags_df = None
post_questions_df = None
post_answers_df = None
merged_df = None
dt_model = None

"""
Data is loaded at once into the dataframe and used from here.
"""
tags_df = pd.DataFrame()


def posts_init():
    global post_questions_df
    global tags_df
    global post_questions_encoded_tags_df
    global post_answers_df
    global post_questions_df
    post_questions_encoded_tags_df = pd.read_csv(Path('./Data/Processed/', 'tags_modified_encoded.csv'))
    post_questions_df = pd.read_csv(Path('./Data/Processed/', 'post_questions.csv'))
    tags_df = pd.read_csv(Path('./Data/Processed/', 'post_questions.csv'))
    post_answers_df = pd.read_csv(Path('./Data/Processed/', 'post_answers.csv'))
    initialize_dataframe()


"""
Question and answer datasets are merged and irrelevant columns are dropped.
"""


def initialize_dataframe():
    global merged_df
    global post_questions_encoded_tags_df
    global post_answers_df

    post_questions_encoded_tags_df = post_questions_encoded_tags_df.fillna(0)
    post_questions_encoded_tags_df.drop("title", axis=1, inplace=True)
    post_questions_encoded_tags_df.drop("body", axis=1, inplace=True)
    post_questions_encoded_tags_df.drop("post_type_id", axis=1, inplace=True)
    post_questions_encoded_tags_df.drop("owner_user_id", axis=1, inplace=True)
    post_answers_df.drop("body", axis=1, inplace=True)

    post_questions_encoded_tags_df['id'] = post_questions_encoded_tags_df['id'].astype(int)
    post_answers_df['parent_id'] = post_answers_df['parent_id'].astype(int)
    merged_df = pd.merge(post_questions_encoded_tags_df, post_answers_df, left_on="id", right_on="parent_id",
                         left_index=False,
                         right_index=False)
    response_time_in_hr = []
    questions_weekday = []
    answers_weekday = []
    years = []
    months = []
    for idx in range(len(merged_df["creation_date_x"])):
        try:
            question_date = datetime.datetime.strptime((merged_df["creation_date_x"].iloc[idx]), '%Y-%m-%d %H:%M:%S.%f+00:00')
        except ValueError:
            new_val = merged_df["creation_date_x"].iloc[idx].split("+")
            merged_df["creation_date_x"].iloc[idx] = new_val[0] + ".0+" + new_val[1]
            question_date = datetime.datetime.strptime((merged_df["creation_date_x"].iloc[idx]), '%Y-%m-%d %H:%M:%S.%f+00:00')
        years.append(question_date.year)
        months.append(question_date.month)
        try:
            answer_date = datetime.datetime.strptime((merged_df["creation_date_y"].iloc[idx]), '%Y-%m-%d %H:%M:%S.%f+00:00')
        except ValueError:
            new_val = merged_df["creation_date_y"].iloc[idx].split("+")
            merged_df["creation_date_y"].iloc[idx] = new_val[0] + ".0+" + new_val[1]
            answer_date = datetime.datetime.strptime((merged_df["creation_date_y"].iloc[idx]), '%Y-%m-%d %H:%M:%S.%f+00:00')
        response_time_in_hr.append((answer_date - question_date).total_seconds() / 3600)
        questions_weekday.append(question_date.weekday())
        answers_weekday.append(answer_date.weekday())

    # Calculating response time in hours to trin model
    merged_df["response_time_in_hr"] = response_time_in_hr
    merged_df["questions_weekday"] = questions_weekday
    merged_df["answers_weekday"] = answers_weekday
    merged_df["year"] = years
    merged_df["month"] = months

    # Dropping irrelevant columns
    merged_df.drop("creation_date_x", axis=1, inplace=True)
    merged_df.drop("creation_date_y", axis=1, inplace=True)
    merged_df.drop("last_activity_date_x", axis=1, inplace=True)
    merged_df.drop("last_activity_date_y", axis=1, inplace=True)
    merged_df.drop("last_edit_date_x", axis=1, inplace=True)
    merged_df.drop("last_edit_date_y", axis=1, inplace=True)
    merged_df.drop("id_x", axis=1, inplace=True)
    merged_df.drop("id_y", axis=1, inplace=True)
    merged_df.drop("comment_count_x", axis=1, inplace=True)
    merged_df.drop("comment_count_y", axis=1, inplace=True)
    merged_df.drop("last_editor_user_id_x", axis=1, inplace=True)
    merged_df.drop("last_editor_user_id_y", axis=1, inplace=True)
    merged_df.drop("score_y", axis=1, inplace=True)

"""
Model is trained based on tags and the day on which question was asked to predict the 
response time for a particular technology. We have used Linear regression as we have label encoded tags 
which made these values continuous and the day of week on which the question was asked was also encoded.
"""


def train_model_for_prediction():
    global post_questions_encoded_tags_df
    global post_answers_df
    global merged_df
    global dt_model
    post_questions_encoded_tags_df.fillna(0)
    post_questions_encoded_tags_df['id'] = post_questions_encoded_tags_df['id'].astype(int)
    post_answers_df['parent_id'] = post_answers_df['parent_id'].astype(int)
    label_encoder = LabelEncoder()
    merged_df["tags_encoded"] = label_encoder.fit_transform(merged_df["tags"])
    features = ["tags_encoded", "questions_weekday"]
    labels = ["response_time_in_hr"]
    X = merged_df[features]
    y = merged_df[labels]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    dt_model = LinearRegression().fit(X_train, y_train)
    pass


"""
This method is responsible to return the prediction response based on user input
"""


def predict(tag, question_weekday):
    global dt_model
    predicted_value = 0
    if len(merged_df.loc[merged_df['tags'] == tag, 'tags_encoded']) > 0:
        data = {'questions_weekday': [question_weekday],
                'tags_encoded': [merged_df.loc[merged_df['tags'] == tag, 'tags_encoded'].iloc[0]]}
        y_test = pd.DataFrame(data)
        predicted_value = dt_model.predict(y_test)[0][0]
        predicted_value=predicted_value/3600
    return round(predicted_value,3)


"""
This method plots the days of week with the percentage distribution of responses received within 1 hour.
"""


def generate_post_bar_chart():
    answered_within_hr = merged_df[merged_df["response_time_in_hr"] <= 1]
    days = merged_df.groupby("answers_weekday").mean().index.get_level_values("answers_weekday").tolist()
    total_answers_per_day_df = pd.DataFrame({'count': merged_df.groupby(["answers_weekday"]).size()}).reset_index()
    total_answers_per_day = total_answers_per_day_df["count"]
    total_answers_per_hr_per_day = answered_within_hr.groupby("answers_weekday").size()
    percent_answers = []
    for idx in range(len(total_answers_per_day)):
        if total_answers_per_hr_per_day.get(idx):
            percent_answers.append(total_answers_per_hr_per_day.get(idx) / total_answers_per_day.get(idx) * 100)
        else:
            percent_answers.append(0)
    total_answers_per_day_df = total_answers_per_day_df.replace(
        {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"})
    # total_answers_per_day_df["answers_weekday"] = merged_df["answers_weekday"].replace(
    #    {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"})
    fig = px.bar(x=total_answers_per_day_df["answers_weekday"], y=percent_answers,
                 labels=dict(x="Weekday", y="Percentage responses on a weekday within an hour"),
                 title="Which day of the week has most questions answered within 1 hour")
    return fig


"""
This method generates the trend in tags/technologies over the years
"""


def generate_tag_trend(tag):
    global merged_df
    merged_df = merged_df.sort_values(by=["year", "month"])
    new_df = merged_df[merged_df["tags"] == tag]
    new_df1 = new_df.groupby(["year"]).size().reset_index(name='count')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=new_df1["year"].unique(), y=new_df1["count"], mode='lines'))
    fig.update_layout(title="Technology trend per year per tag")
    return fig


def gen_mon_year(mergeddata):
    df = mergeddata.copy()

    year = []
    month = []
    for date in df['creation_date'].values:
        try:
            year.append(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f+00:00').year)
            month.append(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f+00:00').month)
        except ValueError:
            new_val = date.split("+")
            date = new_val[0] + ".0+" + new_val[1]
            year.append(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f+00:00').year)
            month.append(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f+00:00').month)

    df['year'] = year
    df['month'] = month

    return df


def generate_line_graph_post_per_year():
    df = gen_mon_year(post_questions_df)

    year_unique = df['year'].unique().tolist()
    year_unique.sort()
    number_of_post = []
    for year in year_unique:
        mask = df['year'] == year
        d = df[mask]
        num = d.shape[0]
        number_of_post.append(num)

    fig = go.Figure(go.Scatter(x=year_unique, y=number_of_post))
    fig.update_layout(
        title="Number of posts added per year",
        xaxis_title="Years",
        yaxis_title="Number of posts added",
        clickmode='event+select'
    )

    return fig
    # return [go.Figure(data=fig)]


def generate_line_graph_post_per_month(year):
    df = gen_mon_year(post_questions_df)

    mask = df['year'] == year

    df = df[mask]

    print(df)

    month_unique = df['month'].unique().tolist()
    month_unique.sort()

    number_of_post = []

    for month in month_unique:
        mask = df['month'] == month
        d = df[mask]
        num = d.shape[0]
        number_of_post.append(num)

    print(month_unique)
    print(number_of_post)

    fig = go.Figure(data=go.Scatter(x=month_unique, y=number_of_post))
    fig.update_layout(
        title="Number of posts added per month",
        xaxis_title="months",
        yaxis_title="Number of posts added",
        clickmode='event+select'
    )

    return fig


#this method will return dataframe with extracting year from date and time formate
def get_year_from_data(data):
    data=tags_df
    data= data.dropna()
    data=data[:5000]
    original_year = []

    # for val in data["creation_date"].values:
    #     try:
    #         original_year.append(datetime.datetime.strptime(val, '%Y-%m-%d %H:%M:%S.%f+00:00').year)
    #     except ValueError:
    #         new_val = val.split("+")
    #         val = new_val[0]+".0+"+new_val[1]
    #         original_year.append(datetime.datetime.strptime(val, '%Y-%m-%d %H:%M:%S.%f+00:00').year)
 
    for idx in range(len(data["creation_date"])):
        try:
            original_year.append(datetime.datetime.strptime(str(data["creation_date"].iloc[idx]), '%Y-%m-%d %H:%M:%S.%f+00:00').year)
        except ValueError:
            new_val = data["creation_date"].iloc[idx].split("+")
            data["creation_date"].iloc[idx] = new_val[0] + ".0+" + new_val[1]
            original_year.append(datetime.datetime.strptime((data["creation_date"].iloc[idx]), '%Y-%m-%d %H:%M:%S.%f+00:00').year)
    data['year']=original_year
    
    return data

#this method will count tags absed on provided year
def generate_tag_years(year):
    idx = 0

    for idx in range(len(tags_df["tags"])):
        tag_value = tags_df["tags"].iloc[idx]
        tag_value = str(tag_value).split("|")
        tags_df["tags"].iloc[idx] = tag_value[0]
        idx += 1

    dataset = get_year_from_data(tags_df)

    dataset = dataset[dataset['year'] == year]
    dataset = dataset.groupby(['year', 'tags'])['tags'].count().reset_index(name='count')
    fig = px.bar(dataset, x="tags", y="count", title="Technologies trend over the years")
    return fig


def generate_table():
    dataset_question = get_year_from_data(post_questions_df)
    question = dataset_question.groupby('year').count().reset_index()
    dataset_answer = get_year_from_data(post_answers_df)
    answer = dataset_answer.groupby('year').count().reset_index()

    fig = go.Figure(data=[go.Table(header=dict(values=['Year', 'Questions', 'Answers']),
                                   cells=dict(values=[question['year'], question['id'], answer['id']]))
                          ])
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Question Anserws Analysis by years",
    )

    return fig
