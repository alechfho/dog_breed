import random

import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model

import tensorflow as tf

from shared_module import *


def input_encoding_model(encoded):
    X_input = Input(encoded)

    X = Dense(4096, activation='sigmoid', name='fc0.0')(X_input)
    X = Dense(2056, activation='sigmoid', name='fc0.1')(X)
    X = Dense(1024, activation='sigmoid', name='fc1.0')(X)
    X = Dense(1024, activation='sigmoid', name='fc1.1')(X)
    X = Dense(512, activation='sigmoid', name='fc2')(X)
    X = Dense(128, activation='sigmoid', name='fc3.0')(X)
    X = Dense(128, activation='sigmoid', name='fc3.1')(X)

    model = Model(inputs=X_input, outputs=X, name='inputEncodingModel')

    return model


def input_training_model(a, p, n, encoding_model):
    X0_Input = Input(a)
    X1_Input = Input(p)
    X2_Input = Input(n)

    X0 = encoding_model(X0_Input)
    X1 = encoding_model(X1_Input)
    X2 = encoding_model(X2_Input)

    model = Model(inputs=[X0_Input, X1_Input, X2_Input], outputs=[X0, X1, X2], name='inputTrainingModel')

    return model


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.square(tf.add(alpha, tf.subtract(pos_dist, neg_dist)))
    loss = tf.reduce_sum(basic_loss)

    return loss


def generate_hard_triplet_dataframe(df_labels, id_column, type_column, n_anchors, model):
    return generate_triplet_dataframe(df_labels, id_column, type_column, n_anchors)


def generate_random_triplet_dataframe(df_labels, id_column, type_column, n_anchors):
    return generate_triplet_dataframe(df_labels, id_column, type_column, n_anchors)


def generate_triplet_dataframe(df_labels, id_column, type_column, n_anchors, model=None):
    unique_types = df_labels[type_column].unique()
    df_anchors = find_all_anchors(df_labels)
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


def find_all_anchors(df_labels):
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
            df_labels['id'] != anchor['id']) & (df_labels['cluster'] != anchor['cluster']), :]


def distance(encoding1, encoding2):
    return np.sum(np.square(np.subtract(encoding1, encoding2)), axis=-1)


def softmax(X):
    ps = np.exp(X)
    ps /= np.sum(ps)
    return ps


def get_class(result, identities):
    resulting_identity = identities.iloc[np.argmax(result)]
    return resulting_identity['breed'], resulting_identity['id'], resulting_identity['encoding']


def model_encode(model, encoding_size):
    def encode(image_encoding):
        encoding = np.zeros((1, 1, encoding_size))
        encoding[0] = image_encoding
        return model.predict_on_batch(encoding)

    return encode


def get_identities_encoding(df_train, encoding_function):
    identities = find_all_anchors(df_train)

    identities_encoding = list(map(lambda x: np.loadtxt(x), identities[ENCODING_COL].values.tolist()))
    identities_encoded = list(map(encoding_function, identities_encoding))
    return identities, identities_encoded


def predict_on_model(df_labels, encoding_function):
    identities, identities_encoded = get_identities_encoding(df_labels, encoding_function)

    prediction_error_count = 0
    prediction = []
    prediction_ids = []
    prediction_encodings = []
    for i, row in df_labels.iterrows():
        row_encoding = encoding_function(np.loadtxt(row.encoding))
        result = []
        for id_encoding in identities_encoded:
            dist = distance(id_encoding, row_encoding)
            result.append(dist.item(0))
        result = softmax(np.array(result))
        result = 1 - result
        predicted_breed, predicted_id, predicted_encoding = get_class(result, identities)
        prediction.append(predicted_breed)
        prediction_ids.append(predicted_id)
        prediction_encodings.append(predicted_encoding)
        if row['breed'] != predicted_breed:
            prediction_error_count += 1

    df_labels['predicted_breed'] = prediction
    df_labels['prediction'] = (df_labels['predicted_breed'] == df_labels['breed'])
    df_labels['prediction_id'] = prediction_ids
    df_labels['prediction_encoding'] = prediction_encodings

    total = df_labels.shape[0]
    bad_predictions = df_labels.loc[df_labels['prediction'] == False].shape[0]
    accuracy = (total - bad_predictions) / total

    return df_labels, total, bad_predictions, accuracy
