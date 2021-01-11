import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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


epoch = 200

results = pd.read_csv('[Amazon] Gain on TOP-10.csv')

# results = results[results['Epoch'] == 200]
x_axes = sorted(results['Alpha'].unique())[1:]
x2 = np.arange(len(x_axes))
accuracy_bprmf = max(results[results['Model'] == 'bprmf']['Recall'])
beyond_accuracy_bprmf = max(results[results['Model'] == 'bprmf']['Novelty'])

scores = []
for epsilon in sorted(results['Epsilon'].unique()):
    if epsilon > 0:
        # We have AMF
        x, y = [], []
        for alpha in x_axes:
            for epoch in sorted(results[(results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (
                    results['Alpha'] == alpha)]['Epoch'].to_list()):
                accuracy_amf = max(
                    results[
                        (results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (results['Alpha'] == alpha) & (
                                    results['Epoch'] == epoch)][
                        'Recall'])
                beyond_accuracy_amf = max(
                    results[
                        (results['Model'] == 'amf') & (results['Epsilon'] == epsilon) & (results['Alpha'] == alpha) & (
                                    results['Epoch'] == epoch)][
                        'Novelty'])
            scores.append([accuracy_amf, beyond_accuracy_amf])
            x.append(accuracy_amf)
            y.append(beyond_accuracy_amf)
        plt.scatter(x, y, label='eps={}'.format(epsilon))

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

plt.plot(x_pareto, y_pareto, color='r')
plt.xlabel('Accuracy')
plt.ylabel('Novelty')

plt.legend()
# plt.show()
plt.savefig('plot_3_pareto.png', format='png')

