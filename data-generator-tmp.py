import pandas
import random
import os


def get_padded_random_float():
    num = random.uniform(0, 1000)
    s = f"{num:.15f}"

    if len(s) < 18:
        s += "0" * (18 - len(s))

    if len(s) > 18:
        s = s[:18]
    
    return s

os.makedirs("data", exist_ok=True)
columns = list(range(random.randint(15, 30)))
nrows = random.randint(500_000, 1_000_000)

df = pandas.DataFrame(
    {
        col: [get_padded_random_float() for _ in range(nrows)]
        for col in columns
    }
)

df.to_csv("data/test1.csv", index=False)

# df = pandas.read_csv("data/test1.csv")
# print(df["0"].mean())
# print(len(df[df["4"] >= 2.710000]))
