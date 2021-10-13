import argparse
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np


def plot(log_dir, plot_type):
    df = pd.read_csv(os.path.join(log_dir, 'progress.csv'))
    data = df[plot_type].values

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(data)),data)
    ax.set_xlabel('episode')
    ax.set_ylabel(plot_type)
    experiment_name = log_dir[log_dir.rindex('/')+1:]
    ax.set_title(experiment_name)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='MaxFinalGoalDistance')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()
    plot(args.log_dir, args.type)