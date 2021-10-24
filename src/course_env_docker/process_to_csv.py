import pandas as pd
import json

def parse(path):
    # g = gzip.open(path, 'rb')
    with open(path) as f:
      for l in f:
          yield json.loads(l)

def getDF(path):
    df = {}
    i = 0
    for d in parse(path):
        df[i] = d
        i += 1
        # if i > 100000:
        #     return pd.DataFrame.from_dict(df, orient='index')

    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('/app/Video_Games_5.json')
df = df[['reviewText', 'overall']]
df = df.dropna()
df = df.drop_duplicates()
df.to_csv('./Video_Games_5.csv')