#!/usr/bin/env bash

python -u train.py --backbone vgg16 > vgg16/logs/train_vgg16_4_22 2>&1
python -u manga_eval.py --backbone vgg16 > vgg16/logs/test_vgg16_4_22 2>&1
python -u train.py --backbone alexnet > alexnet/logs/train_alexnet_4_22 2>&1
python -u manga_eval.py --backbone alexnet > alexnet/logs/test_alexnet_4_22 2>&1
python -u train.py --backbone manga_facenet > manga_facenet/logs/train_manga_facenet_4_22 2>&1
python -u manga_eval.py --backbone manga_facenet > manga_facenet/logs/test_manga_facenet_4_22 2>&1
python -u train.py --backbone sketch_a_net > sketch_a_net/logs/train_sketch_a_net_4_22 2>&1
python -u manga_eval.py --backbone sketch_a_net > sketch_a_net/logs/test_sketch_a_net_4_22 2>&1
