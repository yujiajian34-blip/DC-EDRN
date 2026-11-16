"""Select and verify ordered performance inputs from a locally obtained benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file',required=True,type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'systems/data/evaluation.parquet')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('Refusing to overwrite an existing output.')
    expected = pd.read_csv(ROOT/'systems/data/evaluation_manifest.csv',dtype={'id':str})
    f = pd.read_parquet(args.file,columns=['id','label','split','text'])
    f['id'] = f.id.astype(str)
    if not f.id.is_unique or not set(expected.id).issubset(set(f.id)):
        raise ValueError('Missing or duplicate evaluation IDs.')
    f = f.set_index('id').loc[expected.id].reset_index()
    if len(f) != 3827 or f.text.isna().any(): raise ValueError('Invalid evaluation population.')
    hashes = f.text.map(lambda t:hashlib.sha256(t.encode('utf-8')).hexdigest())
    for actual,wanted in [(hashes,expected.text_sha256),(f.label,expected.label),(f.split,expected.split)]:
        if not (actual.to_numpy() == wanted.to_numpy()).all():
            raise ValueError('Evaluation text/labels/splits differ from archived inputs.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    f.to_parquet(args.output,index=False)
    print(json.dumps({'status':'PASS','rows':len(f),'ordered_text_label_split_identity':True,
                     'training_performed':False,'threshold_fitting':False,
                     'note':'Four inference fields only. Parquet bytes differ from historical full-column data.'},indent=2))

if __name__ == '__main__': main()
