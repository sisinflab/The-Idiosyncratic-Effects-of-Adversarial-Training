import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('[Amazon] Accuracy vs Novelty on TOP-10 - Sheet1.csv')

x_axes = sorted(results['Epoch'].unique())

y_recall_bprmf = []
for x in x_axes:
    y_recall_bprmf.append(results[(results['Model'] == 'BPR-MF') & (results['Epoch'] == x)]['Recall'].values[0])

y_recall_amf = []
for x in x_axes:
    y_recall_amf.append(results[(results['Model'] == 'AMF') & (results['Epoch'] == x)]['Recall'].values[0])

y_novelty_bprmf = []
for x in x_axes:
    y_novelty_bprmf.append(results[(results['Model'] == 'BPR-MF') & (results['Epoch'] == x)]['Novelty'].values[0])

y_novelty_amf = []
for x in x_axes:
    y_novelty_amf.append(results[(results['Model'] == 'AMF') & (results['Epoch'] == x)]['Novelty'].values[0])

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.plot(x_axes, y_recall_bprmf, '--', color='black', label='BPR-MF')
plt.plot(x_axes, y_recall_amf, '-', color='red', label='AMF')
plt.legend()
# plt.show()
plt.savefig('Amazon-Plot-Recall.png', format='png')

# Put Figures Together
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Novelty')
plt.plot(x_axes, y_novelty_bprmf, '--', color='black', label='BPR-MF')
plt.plot(x_axes, y_novelty_amf, '-', color='red', label='AMF')
plt.legend()
# plt.show()
plt.savefig('Amazon-Plot-Novelty.png', format='png')

fig, ax1 = plt.subplots()

color = 'black'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Recall', color=color)
ax1.plot(x_axes, y_recall_bprmf, '--', color=color, label='Recall BPR-MF')
ax1.plot(x_axes, y_recall_amf, '-', color=color, label='Recall AMF')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0)
L1 = plt.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'red'
ax2.set_ylabel('Novelty', color=color)
ax2.plot(x_axes, y_novelty_bprmf, '--', color=color, label='Novelty BPR-MF')
ax2.plot(x_axes, y_novelty_amf, '-', color=color, label='Novelty AMF')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(5)
L2 = plt.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped


# plt.show()
plt.savefig('plot_1_accuracy_vs_beyond.png', format='png')
