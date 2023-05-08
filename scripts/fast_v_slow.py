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
    return [avg1k, avg10k, avg50k, avg100k]

def main() -> None:
    versions = ['cuda', 'default']
    df = pd.DataFrame(columns=versions, 
                      index=['1K', '10K', '50K', '100K'])
    
    for v in versions:
        df[v] = time(v)
        # i just need to see some progress indicator lol
        print(df.head())
        df.to_csv('../data/fastslow_backup.csv', index=False)

    df.rename(columns={'cuda':'optimzed CUDA', 'default':'naive CPU'}, inplace=True)
    fig = df.plot(figsize=(10,10))
    fig.set_xlabel("Number of Bodies")
    fig.set_ylabel("Time Taken (ms)")
    fig = fig.get_figure()
    fig.savefig('../newgraphs/fastslow_cmp.pdf')


if __name__ == "__main__":
    main()