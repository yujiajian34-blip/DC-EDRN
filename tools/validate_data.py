"""Check locally obtained benchmark text against the text-free released manifest."""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def validate(path):
    expected=pd.read_csv(ROOT/'data/split_manifest.csv',dtype={'id':str})
    frame=pd.read_parquet(path)
    required={'id','text','label','split'}
    if not required<=set(frame):raise ValueError('Missing columns: '+str(required-set(frame)))
    frame=frame.copy();frame['id']=frame['id'].astype(str)
    if not frame.id.is_unique or set(frame.id)!=set(expected.id):raise ValueError('Dataset IDs differ from the archived population.')
    frame=frame.set_index('id').loc[expected.id].reset_index()
    actual=frame.text.map(lambda s:hashlib.sha256(str(s).encode('utf-8')).hexdigest())
    if not (actual.to_numpy()==expected.text_sha256.to_numpy()).all():raise ValueError('Text hash mismatch. This is not the archived benchmark.')
    if not (frame.label.to_numpy()==expected.label.to_numpy()).all():raise ValueError('Label mismatch.')
    if not (frame.split.to_numpy()==expected.split.to_numpy()).all():raise ValueError('Split mismatch.')
    return frame,expected

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file',type=Path,required=True)
    p.add_argument('--write-filtered',type=Path,help='Optional output for the fixed-checkpoint supplement; does not reselect checkpoints.')
    a=p.parse_args();frame,expected=validate(a.file)
    clean=frame[~expected.excluded_in_fixed_checkpoint_supplement.to_numpy()].copy()
    if a.write_filtered:
        if a.write_filtered.exists():raise FileExistsError('Refusing to overwrite an existing file: '+str(a.write_filtered))
        a.write_filtered.parent.mkdir(parents=True,exist_ok=True);clean.to_parquet(a.write_filtered,index=False)
    print(json.dumps({'status':'PASS','original_rows':len(frame),'filtered_supplement_rows':len(clean),
          'ids_labels_splits_text_hashes_match':True,'training_performed':False,
          'note':'Checks the reported population/content. Parquet byte hashes can differ by writer version. Other training-target columns are not verified by this text-only manifest.'},indent=2))

if __name__=='__main__':main()
