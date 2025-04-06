from utility.plot_tools import *

matplotlib.use('TkAgg')

import os
import numpy as np
import pandas as pd
import preset

names_model = []
mious_train = []
mious_valid = []

# Iterate over directories and read CSV files
for dirname in os.listdir(preset.dpath_training_records):
    dirpath = os.path.join(preset.dpath_training_records, dirname)

    status = [False, False]
    for fname in os.listdir(dirpath):
        if fname.endswith(".metrics.csv"):
            fpath = os.path.join(dirpath, fname)
            df = pd.read_csv(fpath)
            iou = df['iou'].to_numpy()[-1]
            if fname.endswith("train.metrics.csv"):
                mious_train.append(iou)
                status[0] = True
            elif fname.endswith("valid.metrics.csv"):
                mious_valid.append(iou)
                status[1] = True
    if all(status):
        names_model.append(dirname)

idxs = sorted(list(range(len(names_model))), key=lambda i: mious_valid[i], reverse=True)

names_model = [names_model[i] for i in idxs]
mious_train = [mious_train[i] for i in idxs]
mious_valid = [mious_valid[i] for i in idxs]

# Create the bar plot
X_axis = np.arange(len(names_model))  # Index positions for models

bar_train = plt.barh(X_axis, mious_train, 0.4, label='train', alpha=0.7)  # Draw train bars with some transparency
bar_valid = plt.barh(X_axis, mious_valid, 0.4, label='valid', alpha=0.7)

for bar in bar_train:
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.4f}', va='center', fontsize=8)

for bar in bar_valid:
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.4f}', va='center', fontsize=8)

# plt.axvline(x=1, color='red', linestyle='-', linewidth=2)
# Set y-axis labels to the model names
plt.yticks(X_axis, names_model, rotation=0)

plt.subplots_adjust(left=0.5)
plt.xlabel("miou")
plt.ylabel("Models")
plt.legend()
plt.tight_layout()

plt.show(block=True)
