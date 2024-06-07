
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    
    data = pd.read_csv("Arc-packed-cont.csv")
    mask = data['Step'] <= 20000

    lm_mask = data['med-lm - lm_loss'].notnull() & mask
    lm_data = pd.DataFrame(
        {
            'Step': data['Step'][lm_mask],
            'lm_loss': data['med-lm - lm_loss'][lm_mask]
        }
    )
    lm_roll = lm_data.rolling(window=200).mean()

    arc_mask = (
        data['med-arc-10b - lm_loss'].notnull() &
        data['med-arc-10b - lm_loss_adj'].notnull() & 
        mask
    )
    arc_data = pd.DataFrame(
        {
            'Step': data['Step'][arc_mask],
            'lm_loss': data['med-arc-10b - lm_loss'][arc_mask],
            'lm_loss_adj': data['med-arc-10b - lm_loss_adj'][arc_mask]
        }
    )
    arc_roll = arc_data.rolling(window=200).mean()

    plt.plot(lm_data['Step'], lm_roll['lm_loss'], label='Baseline-LM', color='black')
    plt.plot(arc_data['Step'], arc_roll['lm_loss'], label='MonArc-LM', color='red')
    plt.plot(arc_data['Step'], arc_roll['lm_loss_adj'], label='MonArc-ELM', color='blue')

    # plt.ylim(2.8, 3.5)

    plt.xlabel('Training Step')
    plt.ylabel('NLL Loss [nats/token]')
    plt.title('Loss vs. Fine-Tuning Step (430M Model)')
    plt.legend()
    plt.savefig('finetune-loss-430M.png')


if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()