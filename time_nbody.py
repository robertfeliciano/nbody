#!/usr/bin/env python3
import subprocess
import re
import numpy as np
import seaborn as sns
import pandas as pd

def run(n: int) -> float:
    '''
    Runs the nbody simulation with n bodies for 10 iterations.
    Returns the average time a time iteration took to compute.
    '''
    output = subprocess.run(f"./bin/nbody {n} 10", shell=True, capture_output=True).stdout.decode('utf-8')
    times = np.array([float(t) for t in re.findall(r'\b\d+\.\d\d', output)])
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
    avg300k = run(300000)
    avg3M = run(3000000)
    return [avg1k, avg10k, avg50k, avg300k, avg3M]
    # return [avg1k, avg10k, avg50k]

def main() -> None:
    versions = ['cuda', 'omp', 'basic', 'default']
    df = pd.DataFrame(columns=versions, 
                      index=['1K', '10K', '50K', '300K', '3M'])
                    # index=['1K', '10K', '50K'])
    for v in versions:
        df[v] = time(v)
        # i just need to see some progress indicator lol
        print(df.head())
    # df['cuda'] = time('cuda')
    df.to_csv('backup.csv', index=False)

    fig = df.plot(figsize=(10,10)).get_figure()
    fig.savefig('graph.pdf')


if __name__ == "__main__":
    main()