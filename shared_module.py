ENCODING_COL = 'encoding'
ANP_COL = 'anp'
CLUSTER_COL = 'cluster'

INPUT_DIR = './input'
IMAGE_DIR = '{input}/train-2'.format(input=INPUT_DIR)

ENCODING_SIZE = 4096
VGG19_4096 = 'vgg19_' + str(ENCODING_SIZE)
ANCHORS_10 = 10


def get_path(dataset_name, n_anchors, encoding_strategy):
    def get_file_path(file_type):
        if file_type == 'source':
            return dataset_name + '.csv'
        if file_type == 'encoding':
            return dataset_name + '_' + str(encoding_strategy) + '.csv'
        if file_type == 'anchor':
            return dataset_name + '_anchor_' + str(n_anchors) + '_' + str(encoding_strategy) + '.csv'
        if file_type == 'triplets':
            return dataset_name + '_triplets_' + str(n_anchors) + '_' + str(encoding_strategy) + '.csv'

    return get_file_path
