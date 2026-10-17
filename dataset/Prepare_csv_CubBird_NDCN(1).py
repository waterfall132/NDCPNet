#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare CUB-200-2011 for NDCN.

Differences from the original Prepare_csv_CubBird.py
    1. Split is 100 / 50 / 50 (the protocol your paper states), not 130 / 20 / 50.
    2. Deterministic split by sorted class name, so it is reproducible and
       comparable across machines.
    3. Copies are skipped if the target already exists (safe to re-run).
    4. Aborts on duplicate basenames instead of silently overwriting them.
    5. Optional --bbox to crop to the human-annotated bounding boxes, which is
       what the standard CUB few-shot protocol does. Run it BOTH ways if you
       want the cropped/uncropped comparison for the clutter analysis.

Usage
    python Prepare_csv_CubBird_NDCN.py \
        --data_dir  /data/CUB_200_2011 \
        --save_dir  /data/CUB_200_2011/For_FewShot \
        --bbox
"""
import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', required=True, help='root containing images/ (and bounding_boxes.txt)')
ap.add_argument('--save_dir', required=True)
ap.add_argument('--n_train', type=int, default=100)
ap.add_argument('--n_val', type=int, default=50)
ap.add_argument('--n_test', type=int, default=50)
ap.add_argument('--bbox', action='store_true',
                help='crop to the annotated bounding box (standard CUB few-shot protocol)')
ap.add_argument('--pad', type=float, default=0.0,
                help='fractional padding around the bbox, e.g. 0.1 keeps some context')
args = ap.parse_args()

images_dir = os.path.join(args.data_dir, 'images')
out_img = os.path.join(args.save_dir, 'images')
os.makedirs(out_img, exist_ok=True)

classes = sorted(d for d in os.listdir(images_dir)
                 if os.path.isdir(os.path.join(images_dir, d)))
assert len(classes) == args.n_train + args.n_val + args.n_test, \
    'found %d classes, split asks for %d' % (len(classes),
                                             args.n_train + args.n_val + args.n_test)

# deterministic split: classes are numbered 001..200, take them in order
split_of = {}
for i, c in enumerate(classes):
    if i < args.n_train:
        split_of[c] = 'train'
    elif i < args.n_train + args.n_val:
        split_of[c] = 'val'
    else:
        split_of[c] = 'test'

# ---------------------------------------------------------------- bboxes
bbox = {}
if args.bbox:
    ids, boxes = {}, {}
    with open(os.path.join(args.data_dir, 'images.txt')) as f:
        for line in f:
            i, rel = line.split()
            ids[i] = rel
    with open(os.path.join(args.data_dir, 'bounding_boxes.txt')) as f:
        for line in f:
            p = line.split()
            boxes[p[0]] = tuple(float(v) for v in p[1:5])
    for i, rel in ids.items():
        bbox[rel] = boxes[i]
    print('loaded %d bounding boxes' % len(bbox))

# ---------------------------------------------------------------- copy
seen = defaultdict(list)
rows = {'train': [], 'val': [], 'test': []}
for c in classes:
    split = split_of[c]
    for fn in sorted(os.listdir(os.path.join(images_dir, c))):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        seen[fn].append(c)
        rows[split].append([fn, c])

dupes = {k: v for k, v in seen.items() if len(v) > 1}
if dupes:
    print('ABORT: %d duplicate basenames would collide in the flat images/ dir.' % len(dupes))
    print('       example: %s appears in %s' % (list(dupes)[0], dupes[list(dupes)[0]]))
    sys.exit(1)

n_done = 0
for c in classes:
    for fn in sorted(os.listdir(os.path.join(images_dir, c))):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src = os.path.join(images_dir, c, fn)
        dst = os.path.join(out_img, fn)
        if os.path.exists(dst):
            continue
        if args.bbox:
            rel = '%s/%s' % (c, fn)
            im = Image.open(src).convert('RGB')
            if rel in bbox:
                x, y, w, h = bbox[rel]
                px, py = args.pad * w, args.pad * h
                box = (max(x - px, 0), max(y - py, 0),
                       min(x + w + px, im.width), min(y + h + py, im.height))
                im = im.crop(box)
            im.save(dst, quality=100)
        else:
            shutil.copyfile(src, dst)
        n_done += 1
        if n_done % 1000 == 0:
            print('  %d images written' % n_done)

for split in ['train', 'val', 'test']:
    with open(os.path.join(args.save_dir, '%s.csv' % split), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'label'])
        w.writerows(rows[split])
    print('%-5s : %3d classes, %5d images' % (split, len({r[1] for r in rows[split]}), len(rows[split])))

print('\nDone -> %s   (bbox crop: %s)' % (args.save_dir, args.bbox))
print('Now run:  python check_dataset.py --dataset_dir %s' % args.save_dir)
