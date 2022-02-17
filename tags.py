import pandas as pd
from wordcloud import WordCloud
from pathlib import Path


df = pd.DataFrame()


def tags_init():
    global df
    df = pd.read_csv(Path('./Data/Processed/', 'tags.csv'))


def generate_word_cloud():
    tags_dict = {}

    for idx in range(len(df["tag_name"])):
        tags_dict[df.tag_name[idx]] = df["count"][idx]

    wordcloud = WordCloud(width=1600, height=800, background_color="black", max_words=500).generate_from_frequencies(frequencies=tags_dict)

    return wordcloud.to_image()
