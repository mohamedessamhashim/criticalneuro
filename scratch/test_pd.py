import pandas as pd
s = pd.Series(["seq.7011.8", "seq.10001.7"])
print("Original:")
print(s)
b = s.str.replace(r"^seq\.", "", regex=True)
print("After replace:")
print(b)
c = b.str.split(".")
print("After split:")
print(c)
d = c.str[0]
print("After [0]:")
print(d)

s2 = pd.Series(["SL007011", "SL002564", "SL19233", "SL019233", float("nan")])
s3 = s2.astype(str).str.replace(r"^SL0*", "", regex=True)
print("SomaId stripped:")
print(s3)
