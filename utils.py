import multiprocessing
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tqdm import tqdm

from config import alpha, identity_annot_filename, num_train_samples, lfw_folder, manga_dir, backbone_type


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def get_excludes():
    with open('data/exclude.txt') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


# Get statistics for train data
def get_data_stats():
    with open(identity_annot_filename, 'r') as file:
        lines = file.readlines()

    ids = set()
    images = []
    image2id = {}
    id2images = {}
    excludes = get_excludes()

    for line in lines:
        line = line.strip()
        if len(line) > 0:
            tokens = line.split(' ')
            image_name = tokens[0].strip()
            if image_name not in excludes and image_name != '202599.jpg':
                id = tokens[1].strip()
                ids.add(id)
                images.append(image_name)
                image2id[image_name] = id
                if id in id2images.keys():
                    id2images[id].append(image_name)
                else:
                    id2images[id] = [image_name]

    return list(ids), sorted(images), image2id, id2images


def get_manga_data_stats(usage='train'):

    ids = set()
    images = []
    image2id = {}
    id2images = {}

    for single_manga_dir in os.listdir(os.path.join(manga_dir, usage)):
        if single_manga_dir.startswith('.'):
            continue

        for manga_face in os.listdir(os.path.join(manga_dir, usage, single_manga_dir)):
            if manga_face.startswith('.'):
                continue

            image_path = os.path.join(single_manga_dir, manga_face)
            image_id = manga_face.split('_')[0]
            ids.add(image_id)
            images.append(image_path)
            image2id[image_path] = image_id
            if image_id in id2images.keys():
                id2images[image_id].append(image_path)
            else:
                id2images[image_id] = [image_path]

    return list(ids), sorted(images), image2id, id2images


def triplet_loss(y_true, y_pred):
    a_pred = y_pred[:, 0:128]
    p_pred = y_pred[:, 128:256]
    n_pred = y_pred[:, 256:384]
    positive_distance = K.square(tf.norm(a_pred - p_pred, axis=-1))
    negative_distance = K.square(tf.norm(a_pred - n_pred, axis=-1))
    loss = K.mean(K.maximum(0.0, positive_distance - negative_distance + alpha))
    return loss


def get_random_triplets():
    # Random selection of validation set samples
    ids, images, image2id, id2images = get_data_stats()

    images = images[:num_train_samples]
    num_random_triplets = num_train_samples

    data_set = []

    for i in tqdm(range(num_train_samples)):
        # choose a_image
        while True:
            a_image = random.choice(images)
            a_id = image2id[a_image]
            if len(id2images[a_id]) >= 2: break

        # choose p_image
        while True:
            p_image = random.choice(id2images[a_id])
            if p_image != a_image: break

        # choose n_image
        while True:
            n_image = random.choice(images)
            n_id = image2id[n_image]
            if n_id != a_id: break

        data_set.append({'a': a_image, 'p': p_image, 'n': n_image})

    return data_set


def get_random_manga_triplets(usage='train'):
    # Random selection of validation set samples
    ids, images, image2id, id2images = get_manga_data_stats(usage)

    data_set = []

    for i in tqdm(range(len(images))):
        # choose a_image
        while True:
            a_image = random.choice(images)
            a_id = image2id[a_image]
            if len(id2images[a_id]) >= 2:
                break

        # choose p_image
        while True:
            p_image = random.choice(id2images[a_id])
            if p_image != a_image:
                break

        # choose n_image
        while True:
            n_image = random.choice(images)
            n_id = image2id[n_image]
            if n_id != a_id:
                break

        data_set.append({'a': a_image, 'p': p_image, 'n': n_image})

    return data_set


def get_train_images():
    _, images, _, _ = get_data_stats()
    return images


def get_manga_test_images():
    _, images, _, _ = get_manga_data_stats('test')
    return images


def get_lfw_images():
    with open('data/people.txt', 'r') as file:
        lines = file.readlines()

    names = []

    for i in (range(2, len(lines))):
        line = lines[i].strip()
        tokens = line.split()
        if len(tokens) > 1:
            person_name = tokens[0]
            count = int(tokens[1])
            for j in range(1, count + 1):
                name = '{0}/{0}_{1}.jpg'.format(person_name, str(j).zfill(4))
                filename = os.path.join(lfw_folder, name)
                if os.path.isfile(filename):
                    names.append(name)
                else:
                    raise Exception('File Not Found: {}'.format(filename))

    return names


def get_lfw_pairs():
    with open('data/pairs.txt', 'r') as file:
        lines = file.readlines()

    pairs = []

    for i in (range(1, len(lines))):
        line = lines[i].strip()
        tokens = line.split()
        if len(tokens) == 3:
            person_name = tokens[0]
            id1 = int(tokens[1])
            id2 = int(tokens[2])
            image_name_1 = '{0}/{0}_{1}.jpg'.format(person_name, str(id1).zfill(4))
            image_name_2 = '{0}/{0}_{1}.jpg'.format(person_name, str(id2).zfill(4))
            pairs.append({'image_name_1': image_name_1, 'image_name_2': image_name_2, 'same_person': True})
        elif len(tokens) == 4:
            person_name_1 = tokens[0]
            id1 = int(tokens[1])
            person_name_2 = tokens[2]
            id2 = int(tokens[3])
            image_name_1 = '{0}/{0}_{1}.jpg'.format(person_name_1, str(id1).zfill(4))
            image_name_2 = '{0}/{0}_{1}.jpg'.format(person_name_2, str(id2).zfill(4))
            pairs.append({'image_name_1': image_name_1, 'image_name_2': image_name_2, 'same_person': False})

    return pairs


def get_random_manga_pairs():

    ids, images, image2id, id2images = get_manga_data_stats('test')
    pairs = []

    min_face_num = 100000
    for key, values in id2images.items():
        if len(values) < min_face_num:
            min_face_num = len(values)

    for i in (range(len(ids))):
        label_img_list = id2images[ids[i]][:min_face_num]
        for j in range(len(label_img_list)):
            for k in range(1, len(label_img_list)):
                pairs.append({'image_name_1': label_img_list[j],
                              'image_name_2': label_img_list[k], 'same_person': True})
        for img in label_img_list:
            while True:
                random_img = random.choice(images)
                if image2id[random_img] != image2id[img]:
                    break
            pairs.append({'image_name_1': img,
                          'image_name_2': random_img, 'same_person': False})

    return pairs

    # manga_triplets = get_random_manga_triplets('test')
    # pairs = []
    #
    # for i in (range(len(manga_triplets))):
    #     a_image = manga_triplets[i]['a']
    #     p_image = manga_triplets[i]['p']
    #     n_image = manga_triplets[i]['n']
    #     pairs.append({'image_name_1': a_image, 'image_name_2': p_image, 'same_person': True})
    #     pairs.append({'image_name_1': a_image, 'image_name_2': n_image, 'same_person': False})
    #
    # return pairs


def get_smallest_loss(backbone=None):
    import re
    pattern = 'model.(?P<epoch>\d+)-(?P<val_loss>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    if not backbone:
        backbone = backbone_type
    losses = [float(p.match(f).groups()[1]) for f in os.listdir(backbone + '/models/') if p.match(f)]
    if len(losses) == 0:
        import sys
        return sys.float_info.max
    else:
        return np.min(losses)


def get_latest_model(backbone=None):
    import glob
    import os
    if not backbone:
        backbone = backbone_type
    files = glob.glob(backbone + '/models/*.hdf5')
    files.sort(key=os.path.getmtime)
    if len(files) > 0:
        return files[-1]
    else:
        return None


def get_best_model(backbone=None):
    import re
    pattern = 'model.(?P<epoch>\d+)-(?P<val_loss>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    if not backbone:
        backbone = backbone_type
    if not os.path.exists(backbone + '/models/'):
        os.mkdir(backbone + '/models/')
    files = [f for f in os.listdir(backbone + '/models/') if p.match(f)]
    filename = None
    if len(files) > 0:
        losses = [float(p.match(f).groups()[1]) for f in files]
        best_index = int(np.argmin(losses))
        filename = os.path.join(backbone, 'models', files[best_index])
        print('loading best model: {}'.format(filename))
    return filename
