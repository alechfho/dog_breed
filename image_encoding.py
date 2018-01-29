import os

import numpy as np
from keras import Model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image

from shared_module import *


def try_mkdir(encoding_output_dir):
    try:
        os.mkdir(encoding_output_dir)
    except:
        print(encoding_output_dir + ' already exist')


def encode_images(df_labels, input_dir, image_dir, encoding_strategy):
    encoding_output_dir = input_dir + '/' + encoding_strategy
    try_mkdir(encoding_output_dir)

    df_labels[ENCODING_COL] = df_labels['id'].apply(lambda x: encoding_output_dir + '/' + x + '.txt')

    if encoding_strategy == VGG19_4096:
        vgg19_model = VGG19(weights='imagenet')
        model = Model(inputs=vgg19_model.input, outputs=vgg19_model.get_layer('fc2').output)
        encoding_width = 224
        encoding_height = 224
    for i, rows in df_labels.iterrows():
        img_path = image_dir + rows['id'] + '.jpg'
        encoding_path = '{encoding_output_dir}/{df_label_id}.txt'
        encoding_path = encoding_path.format(encoding_output_dir=encoding_output_dir, df_label_id=rows['id'])
        if os.path.isfile(encoding_path):
            print('Encoding already exist: {encoding_path}. Skipping.'.format(encoding_path=encoding_path))
        else:
            encode_image(img_path, encoding_path, model, encoding_width, encoding_height)

    return df_labels


def encode_image(image_path, encoding_path, model, encoding_width, encoding_height):
    img = image.load_img(image_path, target_size=(encoding_height, encoding_width))
    img_np = image.img_to_array(img)
    img_np = np.expand_dims(img_np, axis=0)
    img_np = preprocess_input(img_np)
    encoding = model.predict(img_np)
    np.savetxt(encoding_path, encoding)
    return
