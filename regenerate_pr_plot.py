import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')

import pickle
from COPY_PASTE_CODE import create_comparison_table, plot_precision_recall_tradeoff

with open('model_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
comparison_df = create_comparison_table(results)
plot_precision_recall_tradeoff(comparison_df.reset_index())
