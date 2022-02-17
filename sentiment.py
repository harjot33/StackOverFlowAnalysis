from re import sub
from users import *
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

vds = SentimentIntensityAnalyzer()

# Data Import
df_questions =  pd.read_csv(Path('./Data/Processed/', 'post_questions.csv'))
df_answers = pd.read_csv(Path('./Data/Processed/', 'post_answers.csv'))
df_comments = pd.read_csv(Path('./Data/Processed/', 'comments.csv'))

# Getting Relevant Data from the Imported Data
df_comlsit  = df_questions["body"].values
df_titlelist = df_questions["title"].values
df_answerbody = df_answers["body"].values
df_commentsbody = df_comments["text"].values
comz = []
title = []
#Filtering and Cleaning User Questions
for com in df_comlsit:
    text = BeautifulSoup(com,features="html.parser").get_text()
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)

    comz.append(text)

for t in df_titlelist:
    text = BeautifulSoup(t,features="html.parser").get_text()
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)
    title.append(text)

answers = []
#Filtering and Cleaning User Answers
for com in df_answerbody:
    text = BeautifulSoup(com,features="html.parser").get_text()
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)

    answers.append(text)

#Filtering and Cleaning User Comments
comments = []
for com in df_commentsbody:
        text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", com)
        text = sub(r"\+", " plus ", text)
        text = sub(r",", " ", text)
        text = sub(r"\.", " ", text)
        text = sub(r"!", " ", text)
        text = sub(r"\?", " ", text)
        text = sub(r"'", " ", text)
        text = sub(r":", " : ", text)
        text = sub(r"\s{2,}", " ", text)

        comments.append(text)


# Adding relevant columns and dropping irrelevant or not needed columns
df_comments["text_clean"] = comments
df_comments.drop(["text","id","post_id","user_id"], inplace=True, axis=1)

df_answers["body_clean"] = answers
df_answers.drop(["body","id","last_activity_date","last_edit_date","last_editor_user_id","parent_id"], inplace=True, axis=1)
df_questions["body_clean"] = comz
df_questions["title_clean"] = title
df_questions["combined"] = df_questions["body_clean"] + df_questions["title_clean"]
columns = ['id','title','body']
df_questions.drop(columns, inplace=True, axis=1)

df_comlsit  = df_questions["combined"].values
comz = []
neg = []
neu = []
pos = []
compound = []
tests = []
# Getting the Sentimental Analysis Scores using Vader
for com in df_comlsit:
    scores = vds.polarity_scores(com)
    neg.append(scores["neg"])
    neu.append(scores["neu"])
    pos.append(scores["pos"])
    compound.append(scores["compound"])

df_questions["neg"] = neg
df_questions["neu"] = neu
df_questions["pos"] = pos
df_questions["compound"] = compound

# Exporting The Dataframe into CSV
df_questions.to_csv('questions.csv')

neg = []
neu = []
pos = []
compound = []
# Getting the Sentimental Analysis Scores using Vader

for com in df_answerbody:
    scores = vds.polarity_scores(com)
    neg.append(scores["neg"])
    neu.append(scores["neu"])
    pos.append(scores["pos"])
    compound.append(scores["compound"])


df_answers["neg"] = neg
df_answers["neu"] = neu
df_answers["pos"] = pos
df_answers["compound"] = compound
# Exporting The Dataframe into CSV

df_answers.to_csv('answers.csv')



neg = []
neu = []
pos = []
compound = []
# Getting the Sentimental Analysis Scores using Vader

for com in df_commentsbody:
    scores = vds.polarity_scores(com)
    neg.append(scores["neg"])
    neu.append(scores["neu"])
    pos.append(scores["pos"])
    compound.append(scores["compound"])


df_comments["neg"] = neg
df_comments["neu"] = neu
df_comments["pos"] = pos
df_comments["compound"] = compound
# Exporting The Dataframe into CSV
df_comments.to_csv('comments.csv')