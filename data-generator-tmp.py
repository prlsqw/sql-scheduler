import pandas
import random
import os


def get_padded_random_float(flt_sz=18):
    num = random.uniform(0, 1000)
    s = f"{num}"

    if len(s) < flt_sz:
        s += "0" * (flt_sz - len(s))

    if len(s) > flt_sz:
        s = s[:flt_sz]
    
    return s

os.makedirs("data", exist_ok=True)
columns = list(range(random.randint(15, 30)))
nrows = random.randint(5_000, 5_000)

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
