#! /usr/bin/env python

import numpy as np

data_path = './trained_models_full/polarity_adv'
pred_file = data_path + '/pred/beams.npz'

data = np.load(pred_file)

preds = data['predicted_ids']
print "Predictions"
print preds[0]

probs = data['log_probs']
print "Probs"
print probs[0]

scores = data['scores']
print "Scores"
print scores[0]
