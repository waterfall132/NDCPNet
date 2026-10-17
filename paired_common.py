"""Shared deterministic Cars training/evaluation protocol."""
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
from PIL import Image
from scipy.stats import t
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_for(*parts):
    return int(digest(parts)[:8], 16)


@contextlib.contextmanager
def image_rng(seed):
    """Isolate CPU augmentation RNG, including when workers=0. Never touch CUDA."""
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def load_split(root, split):
    groups = {}
    with open(Path(root) / (split + '.csv'), newline='') as f:
        for row in csv.DictReader(f):
            groups.setdefault(row['label'], []).append(row['filename'])
    return {k: sorted(v) for k, v in sorted(groups.items())}


def audit_data(root, shot, full=False):
    groups = {s: load_split(root, s) for s in ('train', 'val', 'test')}
    classes = {s: set(g) for s, g in groups.items()}
    counts = [len(groups[s]) for s in groups]
    if counts != [130, 17, 49]:
        raise ValueError(f'This package fixes Cars 130/17/49; got {counts}. Do not regenerate splits.')
    names = set()
    for split, g in groups.items():
        for c, files in g.items():
            if len(files) < shot + 15:
                raise ValueError(f'{split}/{c}: insufficient images')
            for name in files:
                if name in names:
                    raise ValueError(f'Duplicate/cross-split image: {name}')
                names.add(name)
                if full and not (Path(root) / 'images' / name).is_file():
                    raise FileNotFoundError(Path(root) / 'images' / name)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if classes[a] & classes[b]:
            raise ValueError('Class leakage')
    return {s: {'classes': len(g), 'images': sum(map(len, g.values())),
                'csv_sha256': hashlib.sha256((Path(root)/(s+'.csv')).read_bytes()).hexdigest()}
            for s, g in groups.items()}


class Episodes(Dataset):
    def __init__(self, root, split, shot, count, seed, epoch=0):
        self.root, self.split = Path(root), split
        self.seed, self.epoch = seed, epoch
        g = load_split(root, split)
        rng = random.Random(seed_for('tasks', split, seed, epoch, shot))
        self.tasks = []
        for _ in range(count):
            episode = []
            for cls in rng.sample(list(g), 5):
                support = rng.sample(g[cls], shot)
                query = rng.sample([x for x in g[cls] if x not in support], 15)
                episode.append({'class': cls, 'support': support, 'query': query})
            self.tasks.append(episode)
        self.task_hash = digest(self.tasks)
        ops = ([transforms.RandomResizedCrop(84), transforms.ColorJitter(.4, .4, .4),
                transforms.RandomHorizontalFlip()] if split == 'train' else
               [transforms.Resize(92), transforms.CenterCrop(84)])
        self.transform = transforms.Compose(ops + [transforms.ToTensor(),
                                      transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        query, support, labels = [], [], []
        with image_rng(seed_for('augment', self.split, self.seed, self.epoch, idx)):
            for cls, record in enumerate(self.tasks[idx]):
                for kind, dest in [('query', query), ('support', support)]:
                    for name in record[kind]:
                        with Image.open(self.root / 'images' / name) as image:
                            dest.append(self.transform(image.convert('RGB')))
                        if kind == 'query':
                            labels.append(cls)
        return torch.stack(query), torch.stack(support), torch.tensor(labels), idx


def loader(ds, workers):
    # Tasks themselves are randomly ordered. No second shuffle is needed.
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers,
                      pin_memory=True, generator=torch.Generator().manual_seed(731))


def hash_batch(h, q, s, y, idx, lr=None):
    h.update(json.dumps([int(idx), lr]).encode())
    for x in (q, s, y):
        h.update(x.contiguous().numpy().tobytes())


def state_hash(state):
    h = hashlib.sha256()
    for key, v in sorted(state.items()):
        h.update(key.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_hash():
    return digest({str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(BASE.rglob('*.py'))})


def make_model(cfg, arm, device):
    from models.network import define_model
    model = define_model(encoder_model=cfg['encoder'], classifier_model='NDCN',
                         way_num=5, shot_num=cfg['shot'], use_gpu=False,
                         ndcn_cfg=dict(rank=5, n_iter=3, beta=10., margin=.05,
                         div_margin=.3, lambda_div=.2, lambda_mask=.01,
                         mask_budget=.25, pool_floor=.05, score_mode='free',
                         alpha_norm='supminmax', nonneg_mode='relu',
                         use_bg_sep=False, use_qbg=False, use_bgloss=False))
    model.classifier.mask_mode = 'ndm'
    model.classifier.uniform_mask = arm == 'uniform'
    model.classifier.record_diagnostics = False
    return model.to(device)


def ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(x.mean())
    hw = float(t.ppf(.975, n-1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return {'mean': mean, 'ci95_half_width': hw,
            'low': mean-hw if hw is not None else None,
            'high': mean+hw if hw is not None else None, 'tasks': n}


def mask_stats(mask):
    m = mask.detach().double()
    mass = m.sum(1)
    valid = mass > 0
    w = m[valid] / mass[valid, None]
    concentration = w.square().sum(1)
    eff = concentration.reciprocal()
    def mean(v):
        return float(v.mean()) if v.numel() else float('nan')
    return [float(m.mean()), float((~valid).double().mean()),
            mean(concentration), mean(eff), mean(eff / m.shape[1])]


STAT_NAMES = ['mask_mean', 'zero_mask_fraction', 'sum_squared_weights',
              'effective_positions', 'effective_fraction']


@torch.no_grad()
def evaluate(model, ds, device, workers=0, diagnostics=False):
    model.eval()
    model.classifier.record_diagnostics = diagnostics
    scores, predictions, labels, stats_q, stats_s = [], [], [], [], []
    h = hashlib.sha256()
    for q, s, y, idx in loader(ds, workers):
        hash_batch(h, q, s, y, idx)
        out = model(q.to(device, non_blocking=True), s.to(device, non_blocking=True))
        if not torch.isfinite(out).all():
            raise FloatingPointError(f'Nonfinite evaluation logits at task {idx}')
        pred = out.argmax(1).cpu()
        scores.append(int((pred == y).sum()) * 100. / len(y))
        predictions.append(pred.numpy()); labels.append(y.numpy())
        if diagnostics:
            stats_q.append(mask_stats(model.classifier._last_Mq))
            stats_s.append(mask_stats(model.classifier._last_Ms))
    return dict(accuracy=np.asarray(scores), predictions=np.asarray(predictions),
                labels=np.asarray(labels), stats_q=np.asarray(stats_q), stats_s=np.asarray(stats_s),
                input_hash=h.hexdigest(), task_hash=ds.task_hash)


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path, obj):
    atomic(path, lambda f: f.write(json.dumps(obj, indent=2, allow_nan=False).encode()))


def save_checkpoint(path, obj, backup=None, relative=None):
    atomic(path, lambda f: torch.save(obj, f))
    if backup:
        destination = Path(backup) / relative
        atomic(destination, lambda f: copy_into(path, f))


def copy_into(path, dest):
    with open(path, 'rb') as src:
        shutil.copyfileobj(src, dest)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
