import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

df.to_csv("data/wine.csv", index=False)

print("Dataset saved to data/wine.csv")