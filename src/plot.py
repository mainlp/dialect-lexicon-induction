"""
Code for plotting was written with the assistance of Qwen 3 A3B Instruct 2507
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from utils import PROJECT_ROOT


sns.set_context("poster")
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(6, 5))

# Load the data from a CSV file
df = pd.read_csv(f"{PROJECT_ROOT}/results/dialemma_ablation.csv", sep=';')

# Convert perc_train to float
df['perc_train'] = pd.to_numeric(df['perc_train'])

# Convert f1 to float
df['f1'] = pd.to_numeric(df['f1'])

# Exclude average values. Note: A previous version of this file did not include this filter,
# causing one data point to be represented as an outlier at x=0.7.
df = df[df['random_seed'] != 'AVG']

# Create the boxplot
color = 'lightcoral'
linewi = 1
sns.boxplot(
    x='perc_train', y='f1', data=df,
    boxprops=dict(facecolor=color, edgecolor='black', linewidth=linewi, fill=None),
    whiskerprops=dict(color='black', linewidth=linewi),
    capprops=dict(color='black', linewidth=linewi),
    medianprops=dict(color='black', linewidth=linewi),
    ax=ax
)

alpha = 0.2
ax.yaxis.grid(True, alpha=alpha) # Hide the horizontal gridlines
ax.xaxis.grid(True, alpha=alpha) # Show the vertical gridlines
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x}'))

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
xticks = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
xticks = [str(elem*100) + "%" if elem < 0.0100 else str(round(elem*100)) + "%" for elem in xticks]

ax.set_ylim((0.4,0.6))

fontsize = 16
ticksize = 12

# Customize the plot
plt.xlabel('Training set size (%)', fontsize=fontsize)
plt.ylabel('F1 Score', fontsize=fontsize)
plt.xticks(rotation=45, fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.xticks(x, xticks)

# Improve layout
plt.tight_layout()

# Save the plot (optional)
plt.savefig(f"{PROJECT_ROOT}/results/Figure-1.pdf", dpi=300, bbox_inches='tight')

# Show the plot
# plt.show()

# Optional: Print summary statistics
print("\nSummary statistics by perc_train:")
print(df.groupby('perc_train')['f1'].agg(['mean', 'std', 'min', 'max']))
