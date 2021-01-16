import matplotlib.pyplot as plt
import pandas as pd

# results = pd.read_csv('[Amazon] Accuracy vs Novelty on TOP-10 - Sheet1.csv')
results = pd.read_csv('bias_results.csv')

top_k = 50
lr = 0.05

tot_epoch = 200

accuracy_metric = 'Recall'
# 'Precision', 'Recall', 'MAR', 'nDCG',
beyond_accuracy_metric = 'Novelty'
# 'Novelty', 'Coverage', 'Coverage[%]', 'ARP', 'APLT', 'ACLT', 'RSP', 'REO'

x_axes = sorted(results[results['Epoch'] >= (tot_epoch / 2 + 25)]['Epoch'].unique())

results = results[(results['LearnRate'] == lr) & (results['TotEpoch'] == tot_epoch) & (results['Top-K']==top_k)]

epsilon = results[(results['Model']=='amf') & (results[accuracy_metric] == max(results[(results['Model']=='amf')][accuracy_metric]))]['Epsilon'].values[0]
alpha = results[(results['Model']=='amf') & (results[accuracy_metric] == max(results[(results['Model']=='amf')][accuracy_metric]))]['Alpha'].values[0]

y_recall_bprmf = []
for x in x_axes:
    y_recall_bprmf.append(
        results[(results['Model'] == 'bprmf') & (results['Epoch'] == x)][accuracy_metric].values[0])

y_recall_amf = []
for x in x_axes:
    y_recall_amf.append(
        results[(results['Model'] == 'amf') & (results['Epoch'] == x) & (results['Epsilon'] == epsilon) & (
                results['Alpha'] == alpha)][accuracy_metric].values[0])

y_novelty_bprmf = []
for x in x_axes:
    y_novelty_bprmf.append(results[(results['Model'] == 'bprmf') & (results['Epoch'] == x)][beyond_accuracy_metric].values[0])

y_novelty_amf = []
for x in x_axes:
    y_novelty_amf.append(results[
                             (results['Model'] == 'amf') & (results['Epoch'] == x) & (results['Epsilon'] == epsilon) & (
                                     results['Alpha'] == alpha)][beyond_accuracy_metric].values[0])

plt.figure()
plt.xlabel('Epoch')
plt.ylabel(accuracy_metric)
plt.plot(x_axes, y_recall_bprmf, '--', color='black', label='BPR-MF')
plt.plot(x_axes, y_recall_amf, '-', color='red', label='AMF')
plt.legend()
# plt.show()
plt.savefig('./Plot1/Amazon-{0}-Top{1}.png'.format(accuracy_metric, top_k), format='png')

# Put Figures Together
plt.figure()
plt.xlabel('Epoch')
plt.ylabel(beyond_accuracy_metric)
plt.plot(x_axes, y_novelty_bprmf, '--', color='black', label='BPR-MF')
plt.plot(x_axes, y_novelty_amf, '-', color='red', label='AMF')
plt.legend()
# plt.show()
plt.savefig('./Plot1/Amazon-{0}-Top{1}.png'.format(beyond_accuracy_metric, top_k), format='png')

fig, ax1 = plt.subplots()

color = 'black'
ax1.set_xlabel('Epoch')
ax1.set_ylabel(accuracy_metric, color=color)
ax1.plot(x_axes, y_recall_bprmf, '--', color=color, label='{} BPR-MF'.format(accuracy_metric))
ax1.plot(x_axes, y_recall_amf, '-', color=color, label='{} AMF'.format(accuracy_metric))
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0)
L1 = plt.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'red'
ax2.set_ylabel(beyond_accuracy_metric, color=color)
ax2.plot(x_axes, y_novelty_bprmf, '--', color=color, label='{} BPR-MF'.format(beyond_accuracy_metric))
ax2.plot(x_axes, y_novelty_amf, '-', color=color, label='{} AMF'.format(beyond_accuracy_metric))
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(5)
L2 = plt.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# plt.show()
plt.savefig('./Plot1/{0}_vs_{1}_by_epochs-Top{2}.png'.format(accuracy_metric, beyond_accuracy_metric, top_k), format='png')
