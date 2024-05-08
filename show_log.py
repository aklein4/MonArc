
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    
    quasi = pd.read_csv("quasi_log.csv")
    lm = pd.read_csv("lm_log.csv")

    q_roll = quasi["loss"].rolling(window=100).mean()
    lm_roll = lm["loss"].rolling(window=100).mean()

    q_x = np.arange(len(q_roll))[q_roll.notna()]
    lm_x = np.arange(len(lm_roll))[lm_roll.notna()]

    # plot data
    plt.plot(q_x, q_roll[q_roll.notna()], label="quasi")
    plt.plot(lm_x, lm_roll[lm_roll.notna()], label="lm")

    plt.xlim(0, 5000)
    plt.ylim(3.5, 5)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()