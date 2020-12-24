# DATASET
data_path = '../data/{0}/'
InputTrainFile = '../data/{0}/trainingset.tsv'
InputTestFile = '../data/{0}/testset.tsv'
OutputRecResult = '../rec_result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}/'
OutputRecWeight = '../rec_model_weight/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}/'
OutputRecList = '../rec_list/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}/'
output_rec_list_dir = '../rec_list/{0}/'
output_rec_plot_dir = '../rec_plot/{0}/'
output_rec_bias_dir = '../rec_bias/{0}/'
all_items = data_path + 'all_items.csv'
all_interactions = data_path + 'all_interactions.tsv'
users = data_path + 'users.tsv'
items = data_path + 'items.tsv'
training_path = data_path + 'trainingset.tsv'
validation_path = data_path + 'validationset.tsv'
test_path = data_path + 'testset.tsv'
original = data_path + 'original/'
images_path = original + 'images/'
classes_path = original + 'classes_{1}.csv'
dataset_info = data_path + 'stats_after_downloading'
cnn_features_path = original + 'features/cnn_{1}_{2}/'
item_popularity_plot = 'item_popularity_plot.png'
bias_results = 'bias_results.csv'
ttest_bias_results = 'ttest_bias_results.csv'
# RESULTS
weight_dir = 'results/rec_model_weights'
results_dir = 'results/rec_results'

# Fields
user_field = 'user_id'
item_field = 'item_id'
score_field = 'score'
time_field = 'time'


column_order = ['Dataset', 'FileName', 'Model', 'EmbK', 'TotEpoch', 'LearnRate', 'Epsilon', 'Alpha', 'Epoch', 'Top-K', 'Coverage', 'Coverage[%]', 'Precision', 'Recall', 'MAR', 'nDCG', 'Novelty', 'ARP', 'APLT', 'ACLT', 'P_Pop', 'P_Tail', 'RSP', 'PC_Pop', 'PC_Tail', 'REO']
column_order_ttest = ['Dataset', 'FileName', 'Model', 'EmbK', 'TotEpoch', 'LearnRate', 'Epsilon', 'Alpha', 'Epoch', 'Top-K', 'Precision', 'T-Precision', 'Recall', 'T-Recall', 'nDCG', 'T-nDCG',
                      'Novelty', 'T-Novelty', 'ARP', 'T-ARP', 'APLT', 'T-APLT', 'ACLT', 'T-ACLT',
                      'P_Pop', 'T-P_Pop', 'P_Tail', 'T-P_Tail', 'RSP', 'T-RSP', 'PC_Pop', 'T-PC_Pop', 'PC_Tail', 'T-PC_Tail', 'REO', 'T-REO']