import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# results = pd.read_csv('[Amazon] Gain on TOP-10.csv')
results = pd.read_csv('bias_results.csv')

top_k = 50
lr = 0.05
# epsilon = 1.0
# alpha = 0.1
tot_epoch = 200

results = results[(results['LearnRate'] == lr) & (results['TotEpoch'] == tot_epoch) & (results['Top-K']==top_k)]

accuracy_metric = 'Recall'
# 'Precision', 'Recall', 'MAR', 'nDCG',
beyond_accuracy_metric = 'Novelty'
# 'Novelty', 'Coverage', 'Coverage[%]', 'ARP', 'APLT', 'ACLT', 'RSP', 'REO'

x_axes = sorted(results['Alpha'].unique())[1:]

x2 = np.arange(len(x_axes))
accuracy_bprmf = max(results[results['Model'] == 'bprmf'][accuracy_metric])
beyond_accuracy_bprmf = max(results[results['Model'] == 'bprmf'][beyond_accuracy_metric])

for epsilon in sorted(results['Epsilon'].unique()):
    if epsilon > 0:
        # We have AMF
        y_gain = []
        for alpha in x_axes:
            gain = 0
            accuracy_amf = max(
                results[(results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (results['Alpha'] == alpha)][accuracy_metric])
            beyond_accuracy_amf = max(
                results[(results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (results['Alpha'] == alpha)][beyond_accuracy_metric])
            accuracy_gain = (accuracy_amf - accuracy_bprmf) / accuracy_bprmf
            beyond_accuracy_gain = (beyond_accuracy_amf - beyond_accuracy_bprmf) / beyond_accuracy_bprmf
            if accuracy_gain < 0 and beyond_accuracy_gain < 0:
                print('Both Negative in Eps: {} Alhha: {}'.format(epsilon, alpha))
                accuracy_gain *= -1
            gain = accuracy_gain / beyond_accuracy_gain
            # gain = accuracy_gain
            # gain = beyond_accuracy_gain
            y_gain.append(gain)
        plt.scatter(x2, y_gain, label='Eps={}'.format(epsilon))


plt.hlines(0, xmin=x2[0], xmax=x2[-1], linestyles='dashed', colors=['red'])
plt.xlabel('Adversarial Reg. Coeff. (Alpha)')
plt.ylabel('{0} on {1} Gain'.format(accuracy_metric, beyond_accuracy_metric))
plt.xticks(x2, x_axes)
plt.legend()
# plt.show()
plt.savefig('./Plot2/{0}-{1}_gain_by_alpha-Top{2}.png'.format(accuracy_metric, beyond_accuracy_metric, top_k), format='png')

