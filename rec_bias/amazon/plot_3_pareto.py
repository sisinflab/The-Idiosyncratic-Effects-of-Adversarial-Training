import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches


def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


# epoch = 200
#
# results = pd.read_csv('[Amazon] Gain on TOP-10.csv')
results = pd.read_csv('bias_results.csv')

top_k = 50
lr = 0.05
# epsilon = 1.0
# alpha = 0.1
tot_epoch = 200

results = results[(results['LearnRate'] == lr) & (results['TotEpoch'] == tot_epoch) & (results['Top-K'] == top_k)]

accuracy_metric = 'Recall'
# 'Precision', 'Recall', 'MAR', 'nDCG',
beyond_accuracy_metric = 'Novelty'
# 'Novelty', 'Coverage', 'Coverage[%]', 'ARP', 'APLT', 'ACLT', 'RSP', 'REO'

# results = results[results['Epoch'] == 200]
x_axes = sorted(results['Alpha'].unique())[1:]
x2 = np.arange(len(x_axes))

# Pareto on Differences

colors = ['yellow', 'black', 'blue', 'green', 'orange']

for beyond_accuracy_metric in ['Novelty', 'Coverage', 'ARP']:
    plt.figure()

    scores = []
    accuracy_bprmf = max(results[results['Model'] == 'bprmf'][accuracy_metric])
    beyond_accuracy_bprmf = max(results[results['Model'] == 'bprmf'][beyond_accuracy_metric])

    for i, epsilon in enumerate(sorted(results['Epsilon'].unique())):
        if epsilon > 0:
            # We have AMF
            x, y = [], []
            for alpha in x_axes:
                for epoch in sorted(results[(results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (
                        results['Alpha'] == alpha)]['Epoch'].to_list()):
                    accuracy_amf = results[
                            (results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (results['Alpha'] == alpha) & (
                                    results['Epoch'] == epoch)][
                            accuracy_metric].values[0]
                    accuracy_amf = (accuracy_amf - accuracy_bprmf) / accuracy_bprmf

                    beyond_accuracy_amf = results[
                            (results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (results['Alpha'] == alpha) & (
                                    results['Epoch'] == epoch)][
                            beyond_accuracy_metric].values[0]
                    beyond_accuracy_amf = (beyond_accuracy_amf - beyond_accuracy_bprmf) / beyond_accuracy_bprmf

                scores.append([accuracy_amf, beyond_accuracy_amf])
                x.append(accuracy_amf)
                y.append(beyond_accuracy_amf)
            plt.scatter(x, y, label='Epsilon={}'.format(epsilon), color=colors[i])

    scores = np.array(scores)
    pareto = identify_pareto(scores)
    print('Pareto front index vales')
    print('Points on Pareto front: \n', pareto)

    pareto_front = scores[pareto]
    print('\nPareto front scores')
    print(pareto_front)

    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values

    x_pareto = pareto_front[:, 0]
    y_pareto = pareto_front[:, 1]

    plt.plot(x_pareto, y_pareto, '--', color='r')
    plt.xlabel('Delta {}'.format(accuracy_metric))
    plt.ylabel('Delta {}'.format(beyond_accuracy_metric))
    # plt.xlim(-0.1, 0.1)
    # plt.ylim(-0.1, 0.1)

    plt.hlines(0, xmin=-0.25, xmax=0.25, colors=['grey'], )
    plt.vlines(0, ymin=-0.25, ymax=0.25, colors=['grey'])

    plt.legend()

    # currentAxis = plt.gca()
    # currentAxis.add_patch(patches.Rectangle((0, 0), 0.1, 0.1, linewidth=1, edgecolor='r', facecolor='grey', alpha=0.5))

    # plt.show()
    plt.savefig('./Plot3/pareto_{0}-{1}-Top{2}.png'.format(accuracy_metric, beyond_accuracy_metric, top_k), format='png')
