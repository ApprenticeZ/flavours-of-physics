# ensemble method: average voting
# average over multiple submission results
# usage: python avg_vote.py "<list of submission files here, separate with commas>" "output filename"

import numpy as np
import pandas as pd

import sys

in_files = sys.argv[1]
outfile = sys.argv[2]

in_files = in_files.split(',')
print('submission files for average voting: ')
print(in_files)

n_estimator = len(in_files)

t_pred = 0.0
for fname in in_files:
    tmp_submission = pd.read_csv(fname)
    pred = 1.0/n_estimator*tmp_submission["prediction"]
    t_pred = t_pred+pred

submission = pd.DataFrame({"id": tmp_submission['id'], "prediction": t_pred})
submission.to_csv(outfile, index=False)
