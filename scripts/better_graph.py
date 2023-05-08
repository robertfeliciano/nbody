#!/usr/bin/env python3

import pandas as pd


df = pd.read_csv("../data/cuda_backup.csv")
df.rename(columns={"cuda":"optimzed CUDA", "basic":"naive CUDA"}, inplace=True)
# df.to_csv("../data/ftz_backup.csv")
fig = df.plot(figsize=(10,10))
fig.set_xticklabels(['', '1K', '10K', '50K', '100K', '300K', '1M', '3M'])
fig.set_xlabel("Number of Bodies")
fig.set_ylabel("Time Taken (ms)")
fig = fig.get_figure()
fig.savefig('../newgraphs/cuda_cmp.pdf')
