#!/usr/bin/env python3
"""Extract safely; retain existing few-shot CSVs or recover the uploaded DN4 CSVs.
No annotation guessing, random resplitting, renaming by train/test numbering, or JPEG recompression.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from PIL import Image

BASE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with open(path, newline='') as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or []) != {'filename','label'}:
            raise ValueError(f'Unsupported CSV schema: {path}')
        return list(reader)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive', default='/root/StanfordCar.tar.gz')
    p.add_argument('--raw', default='/root/StanfordCars_raw0924')
    p.add_argument('--out', default='/root/StanfordCars_FewShot0924')
    p.add_argument('--csv-dir', help='Explicit directory containing all three intended few-shot CSVs')
    a=p.parse_args()
    archive=Path(a.archive).resolve();raw=Path(a.raw).resolve();out=Path(a.out).resolve()
    if not archive.is_file(): raise FileNotFoundError(archive)
    if out.exists():
        raise FileExistsError(f'{out} exists. Do not overwrite data; check preparation.json or choose a new path.')
    archive_hash=sha(archive)
    marker=raw/'.extraction_complete.json'
    if marker.exists():
        if json.loads(marker.read_text())['archive_sha256']!=archive_hash:
            raise ValueError('Existing extracted directory belongs to another archive')
    else:
        if raw.exists() and any(raw.iterdir()):
            raise FileExistsError('Raw directory nonempty/incomplete; choose a new --raw path')
        raw.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive,'r:*') as t:
            members=t.getmembers()
            for m in members:
                path=PurePosixPath(m.name)
                if path.is_absolute() or '..' in path.parts or not (m.isfile() or m.isdir()):
                    raise ValueError(f'Unsafe/nonregular archive member: {m.name}')
            print(f'Extracting {len(members)} entries to {raw}',flush=True)
            for m in members:
                destination=raw/m.name
                if m.isdir(): destination.mkdir(parents=True,exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if destination.exists(): raise FileExistsError(f'Duplicate archive path: {destination}')
                    with t.extractfile(m) as src, open(destination,'wb') as dst:
                        shutil.copyfileobj(src,dst)
        marker.write_text(json.dumps({'archive_sha256':archive_hash},indent=2))
    candidates=sorted({p.parent for p in raw.rglob('train.csv')
                       if (p.parent/'val.csv').is_file() and (p.parent/'test.csv').is_file()})
    if a.csv_dir: csvdir=Path(a.csv_dir).resolve(); origin='explicit'
    elif len(candidates)==1: csvdir=candidates[0];origin='archive'
    elif len(candidates)>1:
        raise ValueError('Multiple split directories. Specify --csv-dir explicitly: '+str(candidates))
    else: csvdir=BASE/'reference_Cars_csv';origin='previously_uploaded_DN4_Cars_CSVs'
    rows={s:read_rows(csvdir/(s+'.csv')) for s in ('train','val','test')}
    sets={s:{r['label'] for r in rs} for s,rs in rows.items()}
    if [len(v) for v in sets.values()]!=[130,17,49]:
        raise ValueError('Expected Cars 130/17/49 classes. Do not silently replace a different split.')
    if any(sets[x]&sets[y] for x,y in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Class overlap between splits')
    names=[r['filename'] for rs in rows.values() for r in rs]
    if len(names)!=len(set(names)): raise ValueError('Duplicated filename across/within splits')
    index={}
    for path in raw.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}:
            index.setdefault(path.name,[]).append(path)
    resolved={}
    for name in names:
        pth=PurePosixPath(name)
        if pth.is_absolute() or '..' in pth.parts: raise ValueError('Unsafe filename in CSV')
        exact=csvdir/'images'/name
        options=[exact] if exact.is_file() else index.get(pth.name,[])
        if not options:
            raise FileNotFoundError(f'Cannot resolve {name}. Raw cars_train/cars_test archives need an annotation-aware conversion; do not guess numbering.')
        if len(options)>1 and len({sha(v) for v in options})!=1:
            raise ValueError(f'Ambiguous filename {name}: {options}. Not choosing by folder order.')
        resolved[name]=options[0].resolve()
    for i,(name,path) in enumerate(resolved.items()):
        with Image.open(path) as im: im.verify()
        if i%2000==0:print(f'Checked image headers: {i}/{len(resolved)}',flush=True)
    out.mkdir(parents=True)
    for split in rows: shutil.copyfile(csvdir/(split+'.csv'),out/(split+'.csv'))
    for name,path in resolved.items():
        dest=out/'images'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(path)
    record={'archive':str(archive),'archive_sha256':archive_hash,'raw':str(raw),'csv_source':str(csvdir),
            'csv_origin':origin,'class_counts':{s:len(v) for s,v in sets.items()},
            'image_counts':{s:len(rs) for s,rs in rows.items()},
            'csv_sha256':{s:sha(out/(s+'.csv')) for s in rows},
            'matches_archived_csv':{s:sha(out/(s+'.csv'))==sha(BASE/'reference_Cars_csv'/(s+'.csv')) for s in rows},
            'note':'Image links preserve original bytes. Keep raw directory; no JPEG recompression or random class resplit.'}
    (out/'preparation.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2));print('READY:',out)


if __name__=='__main__':main()
