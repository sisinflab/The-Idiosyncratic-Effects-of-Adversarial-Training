import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

results = pd.read_csv('[Amazon] Gain on TOP-10.csv')

# results = results[results['Epoch'] == 200]
x_axes = sorted(results['Alpha'].unique())[1:]
x2 = np.arange(len(x_axes))
accuracy_bprmf = max(results[results['Model'] == 'bprmf']['Recall'])
beyond_accuracy_bprmf = max(results[results['Model'] == 'bprmf']['Novelty'])

for epsilon in sorted(results['Epsilon'].unique()):
    if epsilon > 0:
        # We have AMF
        y_gain = []
        for alpha in x_axes:
            gain = 0
            accuracy_amf = max(
                results[(results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (results['Alpha'] == alpha)][
                    'Recall'])
            beyond_accuracy_amf = max(
                results[(results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (results['Alpha'] == alpha)][
                    'Novelty'])
            accuracy_gain = (accuracy_amf - accuracy_bprmf) / accuracy_bprmf
            beyond_accuracy_gain = (beyond_accuracy_amf - beyond_accuracy_bprmf) / beyond_accuracy_bprmf
            if accuracy_gain < 0 and beyond_accuracy_gain < 0:
                print('Both Negative in Eps: {} Aplha: {}'.format(epsilon, alpha))
                accuracy_gain *= -1
            gain = accuracy_gain / beyond_accuracy_gain
            # gain = accuracy_gain
            # gain = beyond_accuracy_gain
            y_gain.append(gain)
        plt.scatter(x2, y_gain, label='eps={}'.format(epsilon))


plt.hlines(0, xmin=x2[0], xmax=x2[-1], linestyles='dashed', colors=['red'])
plt.xlabel('Adversarial Reg. Coeff. (Alpha)')
plt.ylabel('Accuracy Gain')
plt.xticks(x2, x_axes)
plt.legend()
# plt.show()
plt.savefig('plot_2_gain_by_alpha.png', format='png')

