#!/usr/bin/env python3
import subprocess
import re
import signal
import numpy as np
import pandas as pd

def run(n: int) -> float:
    '''
    Runs the nbody simulation with n bodies for 10 iterations.
    Returns the average time a time iteration took to compute.
    '''
    output = subprocess.run(f"./bin/nbody {n} 10", shell=True, capture_output=True).stdout.decode('utf-8')
    times = np.array([float(t) for t in re.findall(r'\d+\.\d\d', output)])
    avg = round(np.average(times), 3)
    # progress indicator 
    print(f"average for {n} bodies is {avg} ms.")
    return avg

def time_cuda(version: str) -> list[float]:
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

def time_cpu(version: str) -> list[float]:
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
    cuda_versions = ['cuda', 'basic']
    df = pd.DataFrame(columns=cuda_versions, 
                      index=['1K', '10K', '50K', '100K', '300K', '1M', '3M'])
    for v in cuda_versions:
        df[v] = time_cuda(v)
        print(df.head())
        df.to_csv('cuda_backup.csv', index=False)

    fig = df.plot(figsize=(10,10)).get_figure()
    fig.savefig('gpu_cmp.pdf')

    cpu_versions = ['omp', 'default']
    df = pd.DataFrame(columns=cpu_versions, 
                      index=['1K', '10K', '50K', '100K'])
    for v in cpu_versions:
        df[v] = time_cpu(v)
        print(df.head())
        df.to_csv('cpu_backup.csv', index=False)

    fig = df.plot(figsize=(10,10)).get_figure()
    fig.savefig('cpu_cmp.pdf')


if __name__ == "__main__":
    main()