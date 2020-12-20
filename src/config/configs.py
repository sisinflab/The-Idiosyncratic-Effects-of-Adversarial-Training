# DATASET
data_path = 'data/{0}/'
InputTrainFile = 'data/{0}/trainingset.tsv'
InputTestFile = 'data/{0}/testset.tsv'
OutputRecResult = 'rec_result/{0}/{1}_{2}_{3}_{4}_{5}_{6}/'
OutputRecWeight = 'rec_model_weight/{0}/{1}_{2}_{3}_{4}_{5}_{6}/'
OutputRecList = 'rec_list/{0}/{1}_{2}_{3}_{4}_{5}_{6}/'
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

# RESULTS
weight_dir = 'results/rec_model_weights'
results_dir = 'results/rec_results'

# Fields
user_field = 'user_id'
item_field = 'item_id'