"""Fetch three official NeurAlchemy core files and require archived SHA-256 values."""
import argparse,hashlib,json,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',required=True,help='Explicit upstream revision; the historical download revision is unknown.')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'data/DATA_PROTOCOL.json').read_text())
    names={'core_train.parquet':'train','core_validation.parquet':'validation','core_test.parquet':'test'}
    for local,split in names.items():
        target=a.output/local;expected=protocol['raw_core_file_sha256'][local]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError('Existing file hash mismatch: '+local)
            print('VERIFIED existing '+local);continue
        url='https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset/resolve/'+urllib.parse.quote(a.revision,safe='')+'/core/'+split+'-00000-of-00001.parquet'
        request=urllib.request.Request(url,headers={'User-Agent':'DC-EDRN-artifact-reproduction'})
        with urllib.request.urlopen(request,timeout=60) as response:payload=response.read()
        actual=hashlib.sha256(payload).hexdigest()
        if actual!=expected:raise ValueError('Downloaded '+local+' differs from archived input. No output saved. Expected '+expected+'; got '+actual)
        target.write_bytes(payload);print('VERIFIED downloaded '+local)
    print('All three raw-file SHA-256 values match the archived construction inputs. This does not identify when they were originally downloaded.')

if __name__=='__main__':main()
