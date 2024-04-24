import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(file_path):
    df = pd.read_json(file_path, nrows=10, lines=True)
    df = df.drop(['source', 'index', 'question'], axis=1)
    human_df = pd.DataFrame(df["human_answers"].explode("human_answers")).rename(columns={"human_answers": "answer"})
    human_df["is_gpt"] = False

    chatgpt_df = pd.DataFrame(df["chatgpt_answers"].explode("chatgpt_answers")).rename(columns={"chatgpt_answers": "answer"})
    chatgpt_df["is_gpt"] = True

    df = pd.concat([human_df, chatgpt_df], ignore_index=True)

    return train_test_split(df["answer"], df["is_gpt"], test_size=0.2, random_state=42)
