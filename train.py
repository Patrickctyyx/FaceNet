import argparse

import os
import keras
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from config import patience, epochs, backbone_type
from data_generator import MangaDataGenSequence
from model import build_model
from utils import get_available_gpus, get_available_cpus, ensure_folder, triplet_loss, get_smallest_loss, get_best_model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='vgg19', type=str, required=False)
    args = ap.parse_args()
    if args.backbone:
        backbone_type = args.backbone

    checkpoint_models_path = backbone_type + '/models/'
    pretrained_path = get_best_model(backbone_type)
    ensure_folder(backbone_type + '/models/')

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./' + backbone_type + '/logs',
                                               histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', mode='min',
                                       verbose=1, save_best_only=True, save_weights_only=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=int(patience / 2), verbose=1)


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            smallest_loss = get_smallest_loss(backbone_type)
            if float(logs['val_loss']) < smallest_loss:
                self.model_to_save.save(fmt % (epoch, logs['val_loss']))


    # Load our model, added support for Multi-GPUs
    # if using this method you have to recompile the model when you load it
    num_gpu = len(get_available_gpus())
    # num_gpu = 2
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            model = build_model()
            if pretrained_path is not None:
                model = load_model(pretrained_path)

        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        new_model = build_model()
        if pretrained_path is not None:
            new_model = load_model(pretrained_path)

    sgd = keras.optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-6)
    # adam = keras.optimizers.Adam(lr=0.001)
    new_model.compile(optimizer=sgd, loss=triplet_loss)

    print(new_model.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    new_model.fit_generator(MangaDataGenSequence('train'),
                            validation_data=MangaDataGenSequence('validation'),
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=6
                            )
