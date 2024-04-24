import pandas as pd
from sklearn.model_selection import train_test_split

path_df1 = "data/all.jsonl"
path_df2 = "data/data.csv"


def get_data_df1(file_path):
    df = pd.read_json(file_path, lines=True)
    df = df.drop(['source', 'index', 'question'], axis=1)
    human_df = pd.DataFrame(df["human_answers"].explode("human_answers")).rename(columns={"human_answers": "text"})
    human_df["label"] = 1

    chatgpt_df = pd.DataFrame(df["chatgpt_answers"].explode("chatgpt_answers")).rename(columns={"chatgpt_answers": "text"})
    chatgpt_df["label"] = 0

    df = pd.concat([human_df, chatgpt_df], ignore_index=True)
    df["text"] = df["text"].astype(str)

    return df # train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)


def get_data_df2(file_path):
    df = pd.read_csv(file_path)
    return df.drop(['src'], axis=1)


def get_data_mixed():
    df1 = get_data_df1(path_df1)
    df2 = get_data_df2(path_df2)
    return pd.concat([df1, df2], ignore_index=True)
