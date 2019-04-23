import keras.backend as K
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, concatenate, Lambda, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.utils import plot_model

from config import img_size, channel, embedding_size, backbone_type


def build_model(model_type=backbone_type):
    if model_type == 'alexnet':
        base_model = get_alexnet_backbone()
    elif model_type == 'manga_facenet':
        base_model = get_mangafacenet_backbone()
    elif model_type == 'sketch_a_net':
        base_model = get_sketch_a_net_backbone()
    elif model_type == 'inception_resnet_v2':
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, channel),
                                       pooling='avg')
    else:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, channel),
                           pooling='avg')
    image_input = base_model.input
    x = base_model.layers[-1].output
    out = Dense(embedding_size)(x)
    image_embedder = Model(image_input, out)

    input_a = Input((img_size, img_size, channel), name='anchor')
    input_p = Input((img_size, img_size, channel), name='positive')
    input_n = Input((img_size, img_size, channel), name='negative')

    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')

    x = image_embedder(input_a)
    output_a = normalize(x)
    x = image_embedder(input_p)
    output_p = normalize(x)
    x = image_embedder(input_n)
    output_n = normalize(x)

    merged_vector = concatenate([output_a, output_p, output_n], axis=-1)

    model = Model(inputs=[input_a, input_p, input_n],
                  outputs=merged_vector)
    return model


def get_alexnet_backbone():
    main_input = Input((img_size, img_size, channel), name='alexnet_input')

    conv_1 = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), padding='same',
                                    name='conv1')(main_input)
    bn_1 = tf.keras.layers.BatchNormalization()(conv_1)
    relu_1 = tf.keras.layers.ReLU()(bn_1)
    pool_1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(relu_1)

    conv_2 = tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same',
                                    name='conv2')(pool_1)
    bn_2 = tf.keras.layers.BatchNormalization()(conv_2)
    relu_2 = tf.keras.layers.ReLU()(bn_2)
    pool_2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(relu_2)

    conv_3 = tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                                    name='conv3')(pool_2)
    bn_3 = tf.keras.layers.BatchNormalization()(conv_3)
    relu_3 = tf.keras.layers.ReLU()(bn_3)

    conv_4 = tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                                    name='conv4')(relu_3)
    bn_4 = tf.keras.layers.BatchNormalization()(conv_4)
    relu_4 = tf.keras.layers.ReLU()(bn_4)

    conv_5 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                    name='conv5')(relu_4)
    bn_5 = tf.keras.layers.BatchNormalization()(conv_5)
    relu_5 = tf.keras.layers.ReLU()(bn_5)
    pool_5 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool5')(relu_5)

    x = GlobalAveragePooling2D()(pool_5)

    return x


def get_mangafacenet_backbone():
    main_input = Input((img_size, img_size, channel), name='manga_facenet_input')

    layer_1 = tf.keras.layers.Convolution2D(32, (3, 3), padding='same', name='conv1')(main_input)
    # layer_1 = tf.keras.layers.BatchNormalization()(layer_1)
    layer_1 = tf.keras.layers.ReLU()(layer_1)
    layer_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(layer_1)
    layer_1 = tf.keras.layers.Dropout(0.25)(layer_1)

    layer_2 = tf.keras.layers.Convolution2D(64, (3, 3), padding='same', name='conv2')(layer_1)
    # layer_2 = tf.keras.layers.BatchNormalization()(layer_2)
    layer_2 = tf.keras.layers.ReLU()(layer_2)
    layer_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(layer_2)
    layer_2 = tf.keras.layers.Dropout(0.25)(layer_2)

    layer_3 = tf.keras.layers.Convolution2D(128, (3, 3), padding='same', name='conv3')(layer_2)
    # layer_3 = tf.keras.layers.BatchNormalization()(layer_3)
    layer_3 = tf.keras.layers.ReLU()(layer_3)

    layer_4 = tf.keras.layers.Convolution2D(128, (3, 3), padding='same', name='conv4')(layer_3)
    # layer_4 = tf.keras.layers.BatchNormalization()(layer_4)
    layer_4 = tf.keras.layers.ReLU()(layer_4)

    layer_5 = tf.keras.layers.Convolution2D(128, (3, 3), padding='same', name='conv5')(layer_4)
    # layer_5 = tf.keras.layers.BatchNormalization()(layer_5)
    layer_5 = tf.keras.layers.ReLU()(layer_5)
    layer_5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(layer_5)
    layer_5 = tf.keras.layers.Dropout(0.25)(layer_5)

    x = GlobalAveragePooling2D()(layer_5)

    return x


def get_sketch_a_net_backbone():
    main_input = Input((img_size, img_size, channel), name='sketch_a_net_input')

    layer_1 = tf.keras.layers.Convolution2D(64, (15, 15), strides=(3, 3), padding='valid', name='conv1')(main_input)
    # layer_1 = tf.keras.layers.BatchNormalization()(layer_1)
    layer_1 = tf.keras.layers.ReLU()(layer_1)
    layer_1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool1')(layer_1)

    layer_2 = tf.keras.layers.Convolution2D(128, (5, 5), padding='valid', name='conv2')(layer_1)
    # layer_2 = tf.keras.layers.BatchNormalization()(layer_2)
    layer_2 = tf.keras.layers.ReLU()(layer_2)
    layer_2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool2')(layer_2)

    layer_3 = tf.keras.layers.Convolution2D(256, (3, 3), padding='same', name='conv3')(layer_2)
    # layer_3 = tf.keras.layers.BatchNormalization()(layer_3)
    layer_3 = tf.keras.layers.ReLU()(layer_3)

    layer_4 = tf.keras.layers.Convolution2D(256, (3, 3), padding='same', name='conv4')(layer_3)
    # layer_4 = tf.keras.layers.BatchNormalization()(layer_4)
    layer_4 = tf.keras.layers.ReLU()(layer_4)

    layer_5 = tf.keras.layers.Convolution2D(256, (3, 3), padding='none', name='conv5')(layer_4)
    # layer_5 = tf.keras.layers.BatchNormalization()(layer_5)
    layer_5 = tf.keras.layers.ReLU()(layer_5)
    layer_5 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool3')(layer_5)

    x = GlobalAveragePooling2D()(layer_5)

    return x


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)

    K.clear_session()
