import os

img_size = 80
channel = 3
batch_size = 32
triplets_selection_batch_size = 1800
epochs = 15
patience = 10
embedding_size = 128
num_images = 202599
num_identities = 10177
valid_ratio = 0.005
# 5,600 were excluded as they cannot be aligned by dlib
# 202,599 - 5,600 = 196,999, separate into two classes: train and valid.
num_train_samples = 196998
num_lfw_valid_samples = 2185    # LFW data set: 6000 pairs => 2185 triplets
predictor_path = 'models/shape_predictor_5_face_landmarks.dat'
alpha = 0.2
SENTINEL = 1
threshold = 0.8

image_folder = 'data/img_align_celeba'
identity_annot_filename = 'data/identity_CelebA.txt'
bbox_annot_filename = 'data/list_bbox_celeba.txt'
lfw_folder = 'data/lfw_funneled'

semi_hard_mode = 'semi-hard'
hard_mode = 'hard'
triplet_select_mode = hard_mode

best_model = 'models/model.01-0.0087.hdf5'

manga_dir = '/Users/patrick/Documents/datasets/manga109_face'
if not os.path.exists(manga_dir):
    manga_dir = '/home/patrick/manga109_face'
backbone_type = 'vgg16'
# backbone = 'alexnet'
# backbone_type = 'manga_facenet'
# backbone_type = 'sketch_a_net'
# backbone_type = 'inception_resnet_v2'

num_manga_valid_samples = 349  # 349 triplets
