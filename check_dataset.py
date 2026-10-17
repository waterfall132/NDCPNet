#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify a few-shot dataset directory BEFORE launching training.

    python check_dataset.py --dataset_dir /path/to/CUB_birds/For_FewShot

Checks
    1. images/ and train|val|test.csv exist
    2. csv header and row format
    3. every filename listed in the csvs actually exists on disk
    4. no class overlap between splits (would invalidate the protocol)
    5. no duplicate filenames across classes (the flat images/ dir would clobber)
    6. class counts and per-class image counts (must be >= shot + query)
    7. a real image loads and has the expected shape after the transform
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', required=True)
parser.add_argument('--shot_num', type=int, default=5)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--imageSize', type=int, default=84)
args = parser.parse_args()

GREEN, RED, YEL, END = '\033[92m', '\033[91m', '\033[93m', '\033[0m'
n_err = n_warn = 0


def ok(msg):
    print(GREEN + '  [ OK ]  ' + END + msg)


def err(msg):
    global n_err
    n_err += 1
    print(RED + '  [FAIL]  ' + END + msg)


def warn(msg):
    global n_warn
    n_warn += 1
    print(YEL + '  [WARN]  ' + END + msg)


root = args.dataset_dir
print('\nChecking: %s\n' % root)

# ---------------------------------------------------------------- 1. layout
img_dir = os.path.join(root, 'images')
if not os.path.isdir(root):
    err('dataset_dir does not exist'); sys.exit(1)
if os.path.isdir(img_dir):
    n_files = len(os.listdir(img_dir))
    ok('images/ exists (%d files)' % n_files)
else:
    err("images/ not found. The loader builds paths as "
        "<dataset_dir>/images/<filename>, so ALL images must sit flat in one "
        "images/ folder, not in per-class subfolders.")
    sys.exit(1)

splits = {}
for split in ['train', 'val', 'test']:
    p = os.path.join(root, '%s.csv' % split)
    if os.path.isfile(p):
        ok('%s.csv exists' % split)
        splits[split] = p
    else:
        err('%s.csv missing' % split)
if len(splits) < 3:
    sys.exit(1)

# ---------------------------------------------------------------- 2/3. csv
all_classes = {}
all_names = defaultdict(list)
for split, p in splits.items():
    rows = []
    with open(p) as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header != ['filename', 'label']:
            warn("%s.csv header is %s, expected ['filename','label'] "
                 "(the loader skips line 1 regardless, so this is only cosmetic)"
                 % (split, header))
        for line in rd:
            if len(line) != 2:
                err('%s.csv has a malformed row: %s' % (split, line)); break
            rows.append(tuple(line))

    cls = Counter(lbl for _, lbl in rows)
    all_classes[split] = set(cls)
    for fn, lbl in rows:
        all_names[fn].append((split, lbl))

    missing = [fn for fn, _ in rows[:2000] if not os.path.isfile(os.path.join(img_dir, fn))]
    if missing:
        err('%s.csv: %d of the first 2000 files are NOT on disk, e.g. %s'
            % (split, len(missing), missing[:3]))
    else:
        ok('%s.csv: %d images, %d classes, all sampled files found on disk'
           % (split, len(rows), len(cls)))

    too_small = {c: n for c, n in cls.items() if n < args.shot_num + args.query_num}
    if too_small:
        warn('%s.csv: %d class(es) have fewer than shot+query=%d images '
             '(episode_sampling will still run but query sets shrink): %s'
             % (split, len(too_small), args.shot_num + args.query_num,
                list(too_small.items())[:3]))
    else:
        ok('%s.csv: every class has >= shot+query = %d images'
           % (split, args.shot_num + args.query_num))

# ---------------------------------------------------------------- 4. overlap
for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    inter = all_classes[a] & all_classes[b]
    if inter:
        err('%s and %s share %d class(es) -- the label spaces must be disjoint: %s'
            % (a, b, len(inter), sorted(inter)[:3]))
    else:
        ok('%s / %s label spaces are disjoint' % (a, b))

total_cls = sum(len(v) for v in all_classes.values())
print('\n  class split -> train %d | val %d | test %d | total %d'
      % (len(all_classes['train']), len(all_classes['val']),
         len(all_classes['test']), total_cls))

# ---------------------------------------------------------------- 5. dupes
dupes = {fn: v for fn, v in all_names.items() if len(v) > 1}
if dupes:
    err('%d filename(s) appear more than once. Because images/ is flat, the '
        'later copy overwrote the earlier one during preparation and the '
        'labels are now wrong. Example: %s'
        % (len(dupes), list(dupes.items())[0]))
else:
    ok('no duplicate filenames across classes/splits')

# ---------------------------------------------------------------- 7. load one
try:
    from PIL import Image
    import torchvision.transforms as T
    sample = next(iter(all_names))
    im = Image.open(os.path.join(img_dir, sample)).convert('RGB')
    tf = T.Compose([T.Resize(92), T.CenterCrop(args.imageSize), T.ToTensor(),
                    T.Normalize((0.5,) * 3, (0.5,) * 3)])
    t = tf(im)
    if tuple(t.shape) == (3, args.imageSize, args.imageSize):
        ok('sample image loads and transforms to %s' % (tuple(t.shape),))
    else:
        err('unexpected tensor shape %s' % (tuple(t.shape),))
except Exception as e:
    warn('could not run the transform check (%s: %s)' % (type(e).__name__, e))

print('\n' + ('=' * 62))
if n_err:
    print(RED + '%d error(s), %d warning(s) -- fix the errors before training.' % (n_err, n_warn) + END)
    sys.exit(1)
print(GREEN + 'Dataset looks good. %d warning(s).' % n_warn + END)
