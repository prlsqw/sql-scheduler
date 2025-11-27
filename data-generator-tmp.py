import pandas
import random
import os

# os.makedirs("data", exist_ok=True)
# columns = list(range(random.randint(15, 20)))
# nrows = random.randint(500_000, 1_000_000)

# df = pandas.DataFrame(
#     {
#         col: [random.uniform(0, 1000) for _ in range(nrows)]
#         for col in columns
#     }
# )

# df.to_csv("data/test1.csv", index=False)

df = pandas.read_csv("data/test1.csv")
print(df["0"].mean())
print(len(df[df["4"] >= 2.710000]))
