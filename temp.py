import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("cuda_backup.csv")
fig = df.plot(figsize=(10,10))
fig.set_xlabel("Number of Bodies")
fig.set_ylabel("Time Taken (ms)")
fig.set_xticklabels(['0', '1K', '10K', '50K', '100K', '300K', '1M', '3M'])
fig = fig.get_figure()
fig.savefig('cuda_cmp.pdf')

df = pd.read_csv("cpu_backup.csv")
fig = df.plot(figsize=(10,10))
fig.set_xlabel("Number of Bodies")
fig.set_ylabel("Time Taken (ms)")
fig.set_xticklabels(['0', '1K', '', '10K', '', '50K', '', '100K'])
fig = fig.get_figure()
fig.savefig('cpu_cmp.pdf')