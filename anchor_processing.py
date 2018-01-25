import numpy as np
from sklearn.cluster import MiniBatchKMeans

from shared_constants import *


def find_anchors_positives(df_labels, id_column, type_column, n_anchors):
    # get all the unique identities
    unique_ids = df_labels[type_column].unique()
    # for each unique_id cluster the labels and find n_anchors
    ## load all the encodings for each unique_id
    df_labels[ANP_COL] = 'P'
    df_labels[CLUSTER_COL] = -1
    for unique_id in unique_ids:
        ids, encodings, id_encoding = load_encodings_for_type(df_labels, id_column, type_column, unique_id)
        anchor_labels, cluster_results = find_anchor_labels_for_identity(ids, encodings, id_encoding, n_anchors)
        df_labels.loc[df_labels[id_column].isin(anchor_labels), ANP_COL] = 'A'
        df_labels.loc[df_labels[type_column] == unique_id, CLUSTER_COL] = cluster_results

    return df_labels


def load_encodings_for_type(df_labels, id_column, type_column, identity):
    id_encoding = {}
    ids = []
    encodings = []
    type_labels = df_labels[df_labels[type_column] == identity]
    for i, type_label in type_labels.iterrows():
        encoding = np.loadtxt(type_label[ENCODING_COL])
        row_id = type_label[id_column]
        ids.append(row_id)
        encodings.append(encoding)
        id_encoding[row_id] = encoding
    return ids, encodings, id_encoding


def find_anchor_labels_for_identity(ids, encodings, id_encoding, n_anchors):
    kmeans = MiniBatchKMeans(n_clusters=n_anchors, batch_size=20000, random_state=0)
    results = kmeans.fit_predict(encodings)

    identity_labels = find_identity_labels(ids, id_encoding, results, kmeans.cluster_centers_)
    return identity_labels, results


def find_identity_labels(ids, encoding_dict, cluster_result, centroids):
    labels = []

    label_id = 0
    for centroid in centroids:
        result_label = get_nearest_label(ids, encoding_dict, cluster_result, centroid, label_id)
        labels.append(result_label)
        label_id += 1
    return labels


def get_nearest_label(id_labels, encoding_dict, cluster_result, centroid, label_id):
    mask = np.ma.equal(cluster_result, label_id)
    r = np.ma.masked_array(id_labels, mask=np.logical_not(mask))
    r = r.compressed()
    if (r.size > 0):
        return find_closest(r, encoding_dict, centroid)
    else:
        return None


def is_less_than(encoding, closest, centroid):
    d1 = np.sum(np.square(np.subtract(closest, centroid)), axis=-1)
    d2 = np.sum(np.square(np.subtract(encoding, centroid)), axis=-1)
    if d2 <= d1:
        return True
    else:
        return False


def find_closest(id_set, encoding_dict, centroid):
    closest_init = False;
    closest = None
    closest_label = None
    for i in id_set:
        encoding = encoding_dict[i]
        if closest_init == False:
            closest_init = True
            closest = encoding
            closest_label = i
        elif is_less_than(encoding, closest, centroid) == True:
            closest = encoding
            closest_label = i
    return closest_label
