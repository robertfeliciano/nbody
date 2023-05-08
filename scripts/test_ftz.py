#!/usr/bin/env python3
import subprocess
import re
import numpy as np
import pandas as pd

def run(n: int) -> float:
    '''
    Runs the nbody simulation with n bodies for 10 iterations.
    Returns the average time a time iteration took to compute.
    '''
    output = subprocess.run(f"../bin/nbody {n} 10", shell=True, capture_output=True).stdout.decode('utf-8')
    times = np.array([float(t) for t in re.findall(r'\d+\.\d\d', output)])
    avg = round(np.average(times), 3)
    # progress indicator 
    print(f"average for {n} bodies is {avg} ms.")
    return avg

def time(version: str) -> list[float]:
    '''
    Times the version specified by `version` for 5 different body configurations
    '''
    subprocess.run(f"make {version}", shell=True)
    avg1k = run(1000)
    avg10k = run(10000)
    avg50k = run(50000)
    avg100k = run(100000)
    avg300k = run(300000)
    avg1M = run(1000000)
    avg3M = run(3000000)
    return [avg1k, avg10k, avg50k, avg100k, avg300k, avg1M, avg3M]

def main() -> None:
    versions = ['cuda', 'no_ftz']
    df = pd.DataFrame(columns=versions, 
                      index=['1K', '10K', '50K', '100K', '300K', '1M', '3M'])
                    # index=['1K', '10K', '50K'])
    for v in versions:
        df[v] = time(v)
        # i just need to see some progress indicator lol
        print(df.head())
        df.to_csv('../data/ftz_backup.csv', index=False)

    
    df.rename(columns={'cuda':'FTZ', 'no_ftz':'No FTZ'}, inplace=True)
    fig = df.plot(figsize=(10,10))
    fig.set_xlabel("Number of Bodies")
    fig.set_ylabel("Time Taken (ms)")
    fig = fig.get_figure()
    fig.savefig('../newgraphs/test_ftz.pdf')

if __name__ == "__main__":
    main()