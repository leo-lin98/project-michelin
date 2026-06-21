import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("ngshiheng/michelin-guide-restaurants-2021")
print("Path to dataset files:", path)

files = os.listdir(path)
print("Files:", files)

csv_file = [f for f in files if f.endswith(".csv")][0]
michelin_df = pd.read_csv(os.path.join(path, csv_file))

print(michelin_df.head())
print(michelin_df.shape)
print(michelin_df["Award"].unique())

michelin_df["country"] = michelin_df["Location"].apply(
    lambda x: x.split(",")[-1].strip() if "," in str(x) else None
)

stars = ["3 Stars", "1 Star", "2 Stars"]
onlyStars = michelin_df[""]  # TODO: incomplete — column name missing
