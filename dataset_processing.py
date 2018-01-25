import numpy as np
import pandas as pd


def partition_images(df_labels, identifier_label=None, label_postfix='postfix', target_dir='./', filter_identity=[],
                     dev_portion=0.20, encoding_strategy='vgg19_4096'):
    if np.size(filter_identity) == 0:
        filter_identity = df_labels[identifier_label].unique()

    df_filter_labels = df_labels[df_labels.breed.isin(filter_identity)]
    df_filter_identifier_label_count = df_filter_labels.groupby([identifier_label]).agg(['count'])
    df_filter_identifier_label_count['dev_count'] = np.ceil(
        df_filter_identifier_label_count[df_filter_identifier_label_count.columns[0]] * dev_portion).astype(int)

    df_result_train = pd.DataFrame()
    df_result_dev = pd.DataFrame()

    for ident_label, row in df_filter_identifier_label_count.iterrows():
        total = row[0]
        dev_count = row[1]
        train_count = total - dev_count
        df_train, df_dev = filter_images_by_label(df_filter_labels, ident_label, train_count, dev_count)
        df_result_train = df_result_train.append(df_train)
        df_result_dev = df_result_dev.append(df_dev)

    train_label = '{target_dir}/labels_train_{label_postfix}.csv'.format(target_dir=target_dir,
                                                                         label_postfix=label_postfix)
    dev_label = '{target_dir}/labels_dev_{label_postfix}.csv'.format(target_dir=target_dir, label_postfix=label_postfix)

    print('Split into training and dev sets')
    print('Training set in ' + train_label)
    print(df_result_train.groupby([identifier_label]).agg(['count']))
    print('Dev set in ' + dev_label)
    print(df_result_dev.groupby([identifier_label]).agg(['count']))

    df_result_train.to_csv(train_label, index=False)
    df_result_dev.to_csv(dev_label, index=False)
    return


def filter_images_by_label(df_labels, label, train_count, dev_count):
    df_selected_label = df_labels[df_labels.breed.isin([label])]
    df_selected_label_train = df_selected_label.head(train_count)
    df_selected_label_vaidation = df_selected_label.tail(dev_count)
    return df_selected_label_train, df_selected_label_vaidation
