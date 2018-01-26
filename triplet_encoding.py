import numpy as np
import pandas as pd
import random

from shared_constants import *


def generate_hard_triplet_dataframe(df_labels, id_column, type_column, n_anchors, model):
    return generate_triplet_dataframe(df_labels, id_column, type_column, n_anchors)


def generate_random_triplet_dataframe(df_labels, id_column, type_column, n_anchors):
    return generate_triplet_dataframe(df_labels, id_column, type_column, n_anchors)


def generate_triplet_dataframe(df_labels, id_column, type_column, n_anchors, model=None):
    unique_types = df_labels[type_column].unique()
    df_anchors = findAllAnchors(df_labels)
    results = []
    random.seed()
    for anchor_type in unique_types:
        other_types = np.setdiff1d(unique_types, anchor_type)
        anchors = find_anchors(df_labels, anchor_type, type_column, n_anchors)
        for i, anchor in anchors.iterrows():
            positives = find_positives(df_labels, anchor, anchor_type, type_column)
            negatives = find_negatives(df_labels, other_types, type_column)
            for ni, n in negatives.iterrows():
                if model is None:
                    selected_positives = select_positives(anchor, positives)
                for p in selected_positives:
                    results.append(
                        [anchor[id_column], anchor[type_column], anchor[ENCODING_COL], p[id_column], p[type_column],
                         p[ENCODING_COL], n[id_column], n[type_column], n[ENCODING_COL]])
    return df_anchors, pd.DataFrame(results,
                                    columns=['anchor_id', 'anchor_type', 'anchor_encoding',
                                             'positive_id', 'positive_type', 'positive_encoding',
                                             'negative_id', 'negative_type', 'negative_encoding'])


def select_positives(anchor, positives):
    selected = random.randint(0, positives.shape[0] - 1)
    return [positives.iloc[selected]]


def findAllAnchors(df_labels):
    df_anchors = df_labels.loc[df_labels[ANP_COL] == 'A']
    return df_anchors


def generate_triplet(df_labels, type_column, n_anchors):
    unique_types = df_labels[type_column].unique()
    results = []
    for anchor_type in unique_types:
        other_types = np.setdiff1d(unique_types, anchor_type)
        anchors = find_anchors(df_labels, anchor_type, type_column, n_anchors)
        for i, anchor in anchors.iterrows():
            random.seed()
            positives = find_positives(df_labels, anchor, anchor_type, type_column)
            negatives = find_negatives(df_labels, other_types, type_column)
            anchor_dict = {'id': anchor['id'], type_column: anchor[type_column], ENCODING_COL: anchor[ENCODING_COL]}
            for ni, n in negatives.iterrows():
                neg_dict = {'id': n['id'], type_column: n[type_column], ENCODING_COL: n[ENCODING_COL]}
                p1 = positives.iloc[random.randint(0, positives.shape[0] - 1)]
                pos_dict = {'id': p1['id'], type_column: p1[type_column], ENCODING_COL: p1[ENCODING_COL]}
                results.append((anchor_dict, pos_dict, neg_dict))
    return np.array(results)


def find_anchors(df_labels, anchor_type, type_column, n_anchors):
    anchors = df_labels.loc[(df_labels[type_column] == anchor_type) & (df_labels[ANP_COL] == 'A'), :]
    if anchors.shape[0] < n_anchors:
        print('Asking for n_achors {n_anchors} but only have {actual} from dataframe for type {anchor_type}'.format(
            n_anchors=n_anchors, actual=anchors.shape[0], anchor_type=anchor_type))
    return anchors.head(n_anchors)


def find_negatives(df_labels, other_types, type_column):
    return df_labels.loc[(df_labels[type_column].isin(other_types))]


def find_positives(df_labels, anchor, anchor_type, type_column):
    return df_labels.loc[(df_labels[type_column] == anchor_type) & (df_labels[ANP_COL] == 'P') & (
            df_labels['id'] != anchor['id']), :]
