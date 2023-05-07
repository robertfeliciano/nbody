#!/usr/bin/env python3

import pandas as pd


df = pd.read_csv("../data/fastslow_backup.csv")
# df.rename(columns={"cuda":"with ftz", "no_ftz":"no ftz"}, inplace=True)
# df.to_csv("../data/ftz_backup.csv")
fig = df.plot(figsize=(10,10))
fig.set_xlabel("Number of Bodies")
fig.set_ylabel("Time Taken (ms)")
fig.set_xticklabels(['0', '1K', '', '10K', '', '50K', '', '100K'])
fig = fig.get_figure()
fig.savefig('../graphs/fastslow_cmp.pdf')
