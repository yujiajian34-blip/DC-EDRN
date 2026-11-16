from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy, hashlib, importlib.util, json, platform, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ispa_matched', ROOT/'code/vendor/MatchedLocalEncoderBaselines-v1.py')
MATCHED = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = MATCHED
spec.loader.exec_module(MATCHED)
V7 = MATCHED.V7
GROUPS = {'core':['DC_EDRN','BiLSTM','ByteCNN'],
          'ablations':['DC_NoSpanGate','DC_NoWord','WordOnly'],
          'extended':['Full_EDRN','BiGRU','TinyTransformer']}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def dump(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(temp,path)

def save_torch(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    torch.save(data,temp);os.replace(temp,path)

def environment(device):
    return {'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
            'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
            'cuda_runtime':torch.version.cuda,'device':str(device),
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'cpu_threads':torch.get_num_threads()}

def get_device(choice):
    if choice=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate your CUDA-enabled PyTorch environment; use --device cpu only for smoke tests.')
    return torch.device(choice)

class WordOnly(nn.Module):
    """Lexical baseline: same hashed encoder; binary + intent heads; no byte path."""
    def __init__(self,c):
        super().__init__()
        self.word_encoder=V7.HashedWordContextEncoder(c['word_vocab'],c['word_dim'],c['semantic_dim'],c['semantic_kernels'],c['dropout'])
        dim=self.word_encoder.output_dim
        self.classifier=nn.Sequential(V7.RMSNorm(dim),nn.Linear(dim,c['units']),nn.GELU(),nn.Dropout(c['dropout']),nn.Linear(c['units'],1))
        self.intent_head=nn.Linear(dim,3)
        self.intent_scale_raw=nn.Parameter(torch.tensor(-1.0))
    def forward(self,input_ids,word_ids):
        w=self.word_encoder(word_ids);intent=self.intent_head(w)
        logits=self.classifier(w).squeeze(-1)+F.softplus(self.intent_scale_raw)*(intent[:,2]-torch.logsumexp(intent[:,:2],1))
        return {'logits':logits,'intent_logits':intent,'span_logits':None,'duration_logits':None,'gate_logits':None,'mask':None}

def model(c):
    return WordOnly(c) if c.get('word_only') else MATCHED.MatchedLocalEncoderModel(c,4)

def loss(outputs,batch,args,positive_weight):
    if outputs['span_logits'] is not None:
        return V7.compute_loss(outputs,batch,args,positive_weight)[0]
    classification=(F.binary_cross_entropy_with_logits(outputs['logits'],batch[1],pos_weight=positive_weight,reduction='none')*batch[4]).mean()
    return classification+args.intent_loss_weight*F.cross_entropy(outputs['intent_logits'],batch[5])

class ZeroDuration(nn.Module):
    def forward(self,pooled):return pooled.new_zeros((len(pooled),3))

def inference_copy(m):
    m=copy.deepcopy(m).eval()
    if hasattr(m,'duration_head'):m.duration_head=ZeroDuration()
    return m

def parameters(m):return sum(p.numel() for p in m.parameters())

def metrics(labels,scores,threshold):
    result=V7.calculate_metrics(labels,scores,threshold)
    # A positive-only split has undefined FPR; do not turn it into a measured zero.
    if not np.any(np.asarray(labels)==0):result['FPR']=float('nan')
    if not np.any(np.asarray(labels)==1):result['FNR']=float('nan')
    return result

def select_threshold(y,p,ni,od,ni_cap=.10,od_cap=.15):
    """Exact sorted-count sweep; >= decision rule; validation inputs only."""
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    ni=np.asarray(ni,dtype=float);od=np.asarray(od,dtype=float)
    if not len(ni) or not len(od) or set(np.unique(y))!={0,1}:raise ValueError('Selection needs both validation classes and both benign validation sets.')
    if not all(np.isfinite(x).all() for x in (p,ni,od)):raise ValueError('Nonfinite model scores.')
    thresholds=np.unique(np.r_[0.,1.,np.nextafter(1.,np.inf),p,ni,od])
    pos=np.sort(p[y==1]);neg=np.sort(p[y==0])
    tp=len(pos)-np.searchsorted(pos,thresholds,side='left')
    fp=len(neg)-np.searchsorted(neg,thresholds,side='left');fn=len(pos)-tp
    f1=2*tp/np.maximum(2*tp+fp+fn,1)
    nf=(len(ni)-np.searchsorted(np.sort(ni),thresholds,side='left'))/len(ni)
    of=(len(od)-np.searchsorted(np.sort(od),thresholds,side='left'))/len(od)
    violation=np.maximum(0,nf-ni_cap)+np.maximum(0,of-od_cap)
    feasible=violation<=1e-12;score=f1-2*violation
    best=np.lexsort((-thresholds,f1,score,feasible.astype(int)))[-1]
    return {'threshold':float(thresholds[best]),'f1':float(f1[best]),'ni_fpr':float(nf[best]),
            'od_fpr':float(of[best]),'feasible':bool(feasible[best]),'score':float(score[best])}

@torch.inference_mode()
def predict(m,ds,device,batch_size=256):
    loader=V7.make_loader(ds,batch_size,False,42,0);values=[]
    m.eval()
    # FP32 scoring for stable threshold selection, independent of training AMP.
    for b in loader:
        values.extend(torch.sigmoid(m(b[0].to(device),b[6].to(device))['logits']).float().cpu().tolist())
    values=np.asarray(values)
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite evaluation scores.')
    return values

def prepared():
    p=ROOT/'data/prepared/benchmark_known_matches_filtered.parquet'
    if not p.exists():raise FileNotFoundError('Run: python code/prepare_data.py')
    return p

def encode_texts(texts,max_bytes=512,max_words=128,vocab=8192):
    byte=np.zeros((len(texts),max_bytes),dtype=np.int64);word=np.zeros((len(texts),max_words),dtype=np.int64)
    for i,text in enumerate(texts):
        raw=str(text).encode('utf-8')[:max_bytes]
        byte[i,:len(raw)]=np.frombuffer(raw,dtype=np.uint8).astype(np.int64)+1
        for j,t in enumerate(V7.WORD_PATTERN.findall(str(text).casefold())[:max_words]):
            word[i,j]=V7.zlib.crc32(t.encode('utf-8'))%(vocab-1)+1
    return torch.from_numpy(byte),torch.from_numpy(word)
