import pandas as pd
from transformers import pipeline

pipe = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-qa-detector-roberta")
df = pd.read_json("data/all.jsonl", lines=True, nrows=10)

df.to_json("test")

human_df = df["human_answers"]
chatgpt_df = df["chatgpt_answers"]

count_human = 0
count_chatgpt = 0
for i in range(10):
    print(human_df[i])
    print(len(human_df[i]))
    print(len(chatgpt_df[i]))
    inference_human = pipe(human_df[i])
    inference_chatgpt = pipe(chatgpt_df[i])
    if inference_human[0]["label"] == "LABEL_0":
        count_human += 1
    if inference_chatgpt[0]["label"] == "LABEL_1":
        count_chatgpt += 1
print(count_human / 10)
print(count_chatgpt / 10)