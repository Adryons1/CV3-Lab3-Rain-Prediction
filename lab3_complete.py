"""Computer Vision 3 - Laborator 3. Experimente reproductibile PyTorch.
Date: weatherAUS.csv. Ruleaza in Colab sau local, dupa instalarea dependentelor.
Split-uri stratificate fixe; evaluarea pe test se realizeaza doar la final.
"""
from __future__ import annotations
import os, time, random, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
import matplotlib.pyplot as plt

RANDOM_SEED = 1
BASE6 = ['MinTemp','MaxTemp','Rainfall','Humidity3pm','Pressure9am','RainToday']
EXTRA2 = ['WindSpeed3pm','Humidity9am']
FEATURES = BASE6 + EXTRA2
TARGET = 'RainTomorrow'
BATCH_SIZE = 1024
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_all(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_data(path):
    df = pd.read_csv(path)
    d = df[FEATURES + [TARGET]].copy()
    d['RainToday'] = d['RainToday'].map({'No': 0., 'Yes': 1.})
    d[TARGET] = d[TARGET].map({'No': 0, 'Yes': 1})
    d = d.dropna(subset=[TARGET]); d[TARGET] = d[TARGET].astype(int)
    full = d.reset_index(drop=True)
    cc = full.dropna(subset=FEATURES).reset_index(drop=True)
    return full, cc

def split(df, seed=RANDOM_SEED):
    ids = np.arange(len(df)); y = df[TARGET].to_numpy().astype(int)
    trv, te = train_test_split(ids, test_size=.15, random_state=seed, stratify=y)
    tr, va = train_test_split(trv, test_size=.15/.85, random_state=seed, stratify=y[trv])
    return tr,va,te

def xy(df, ids, features):
    z=df.iloc[ids]
    return z[features].to_numpy(dtype=np.float32), z[TARGET].to_numpy(dtype=np.int64)

def scale_arrays(xtr,xva,xte=None):
    sc=StandardScaler().fit(xtr)
    a=sc.transform(xtr).astype('float32');b=sc.transform(xva).astype('float32')
    return (a,b,sc) if xte is None else (a,b,sc.transform(xte).astype('float32'),sc)

class MLP(nn.Module):
    def __init__(self, n_features, hidden=(64,32), dropout=0., activation='ReLU'):
        super().__init__()
        activation_cls={'ReLU':nn.ReLU,'Tanh':nn.Tanh}[activation]
        layers=[]; in_dim=n_features
        for h in hidden:
            layers.extend([nn.Linear(in_dim,h),activation_cls()])
            if dropout: layers.append(nn.Dropout(dropout))
            in_dim=h
        layers.append(nn.Linear(in_dim,1))
        self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x).squeeze(-1)

def calc_metrics(y,probs,threshold=.5):
    yp=(probs>=threshold).astype('int64')
    p,r,f,_=precision_recall_fscore_support(y,yp,labels=[0,1],zero_division=0)
    return dict(acc=float(accuracy_score(y,yp)),p0=float(p[0]),r0=float(r[0]),f0=float(f[0]),p1=float(p[1]),r1=float(r[1]),f1=float(f[1]))

def fit_mlp(xtr,ytr,xva,yva,hidden=(64,32),epochs=EPOCHS,batch_size=BATCH_SIZE,lr=.001,optimizer='Adam',weight_decay=0.,l1_lambda=0.,dropout=0.,activation='ReLU',device=None,seed=RANDOM_SEED):
    seed_all(seed)
    dev=device or DEVICE
    model=MLP(xtr.shape[1],hidden,dropout,activation).to(dev)
    train_x=torch.tensor(xtr,dtype=torch.float32,device=dev)
    train_y=torch.tensor(ytr,dtype=torch.float32,device=dev)
    val_x=torch.tensor(xva,dtype=torch.float32,device=dev)
    val_y=torch.tensor(yva,dtype=torch.float32,device=dev)
    crit=nn.BCEWithLogitsLoss()
    opts={'Adam':lambda: torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay),
          'SGD':lambda: torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=weight_decay),
          'RMSprop':lambda: torch.optim.RMSprop(model.parameters(),lr=lr,weight_decay=weight_decay)}
    opt=opts[optimizer]()
    hist={'train_loss':[],'val_loss':[],'val_acc':[]}
    best=float('inf');best_epoch=None;best_state=None;best_metrics=None
    t0=time.perf_counter();n=len(train_y)
    for ep in range(1,epochs+1):
        model.train(); perm=torch.randperm(n,device=dev);sum_loss=0.
        for idx in perm.split(batch_size):
            opt.zero_grad(set_to_none=True)
            loss=crit(model(train_x[idx]),train_y[idx])
            if l1_lambda:
                loss=loss+l1_lambda*sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name)
            loss.backward();opt.step()
            sum_loss+=float(loss.detach().item())*len(idx)
        model.eval()
        with torch.no_grad():
            logits=model(val_x)
            vl=float(crit(logits,val_y).item())
            probs=torch.sigmoid(logits).cpu().numpy()
        met=calc_metrics(yva,probs)
        hist['train_loss'].append(sum_loss/n);hist['val_loss'].append(vl);hist['val_acc'].append(met['acc'])
        if vl<best:
            best=vl;best_epoch=ep;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_metrics=met
    if dev.type=='cuda':torch.cuda.synchronize()
    secs=time.perf_counter()-t0
    model.load_state_dict(best_state)
    return model,dict(best_val_loss=best,best_epoch=best_epoch,seconds=secs,**best_metrics),hist

def result(label,row,**other):
    return dict(experiment=label,**other,**row)

def do_all(csv_path='weatherAUS.csv',out_dir='lab3_results',run_benchmark=True):
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    seed_all()
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    full,cc=load_data(csv_path)
    tr,va,te=split(cc)
    xc_tr,ytr=xy(cc,tr,FEATURES); xc_va,yva=xy(cc,va,FEATURES)
    x8tr,x8va,sc8=scale_arrays(xc_tr,xc_va)
    rows={}; hists={}
    def register(group,label,meta,hist):
        rows.setdefault(group,[]).append(result(label,meta));hists.setdefault(group,{})[label]=hist
        print(f'{group} | {label}: val loss={meta["best_val_loss"]:.4f}, acc={meta["acc"]:.4f}, F1 ploaie={meta["f1"]:.4f}, ep={meta["best_epoch"]}',flush=True)
    # TODO1: comparatie pe aceleasi randuri complete
    x6tr,x6va,_=scale_arrays(*[xy(cc,i,BASE6)[0] for i in (tr,va)])
    for label,a,b in [('6 descriptori',x6tr,x6va),('8 descriptori',x8tr,x8va)]:
        model,met,h=fit_mlp(a,ytr,b,yva)
        register('features',label,met,h)
        if label=='8 descriptori':
            first=next(l for l in model.modules() if isinstance(l,nn.Linear))
            rows['weights']=[{'feature':f,'mean_abs_weight':float(w)} for f,w in zip(FEATURES,first.weight.detach().cpu().abs().mean(0))]
    # TODO2: imputation learned strictly on train
    a_tr,a_va,a_te=split(full)
    raw_tr,yt=xy(full,a_tr,FEATURES);raw_va,yv=xy(full,a_va,FEATURES)
    keeptr=np.isfinite(raw_tr).all(axis=1);keepva=np.isfinite(raw_va).all(axis=1)
    tr_cc=raw_tr[keeptr]; yt_cc=yt[keeptr]
    val_common=raw_va[keepva]; yval_common=yv[keepva]
    for name, train_data, labels in [('Stergere randuri NaN',tr_cc,yt_cc),('Imputare medie train',raw_tr,yt)]:
        if name.startswith('Imputare'):
            imp=SimpleImputer(strategy='mean').fit(train_data)
            tt=imp.transform(train_data).astype(np.float32)
        else:tt=train_data
        xtr1,xva1,_=scale_arrays(tt,val_common)
        _,met,h=fit_mlp(xtr1,labels,xva1,yval_common,epochs=15)
        register('missing',name,dict(n_train=int(len(labels)),n_val=int(len(yval_common)),**met),h)
    # TODO3: class ratio and oversampling only on train, validation untouched
    idx0=np.where(ytr==0)[0];idx1=np.where(ytr==1)[0];ng=np.random.default_rng(RANDOM_SEED)
    ratios=[('3.5:1',3.5),('3:1',3.),('2:1',2.),('Over-sampling 1:1',1.)]
    for name,ratio in ratios:
        if name.startswith('Over'):
            sampled1=ng.choice(idx1,size=len(idx0),replace=True)
            ix=np.concatenate([idx0,sampled1])
        else:
            selected0=ng.choice(idx0,size=min(len(idx0),int(round(ratio*len(idx1)))),replace=False)
            ix=np.concatenate([selected0,idx1])
        ix=ng.permutation(ix)
        _,met,h=fit_mlp(x8tr[ix],ytr[ix],x8va,yva,epochs=15)
        register('imbalance',name,dict(n_train=int(len(ix)),ratio=float(np.sum(ytr[ix]==0)/np.sum(ytr[ix]==1)),**met),h)
    # TODO4: minimum 5 optimizations; selection on validation only
    configs={
        'MLP 64-32 (referinta)':dict(hidden=(64,32)),
        'MLP 32-16':dict(hidden=(32,16)),
        'MLP 128-64-32':dict(hidden=(128,64,32)),
        'MLP 64-32 + Dropout 0.2':dict(hidden=(64,32),dropout=.2),
        'MLP 64-32 + L2 0.001':dict(hidden=(64,32),weight_decay=.001),
        'MLP 64-32 + Tanh':dict(hidden=(64,32),activation='Tanh'),
        'MLP 64-32 LR 0.0003':dict(hidden=(64,32),lr=.0003),
        'MLP 64-32 LR 0.003':dict(hidden=(64,32),lr=.003),
        'MLP 64-32 SGD LR 0.01':dict(hidden=(64,32),optimizer='SGD',lr=.01),
        'MLP 64-32 RMSprop LR 0.001':dict(hidden=(64,32),optimizer='RMSprop',lr=.001)
    }
    for name,cfg in configs.items():
        _,met,h=fit_mlp(x8tr,ytr,x8va,yva,epochs=18,**cfg)
        register('selection',name,met,h)
    selection=pd.DataFrame(rows['selection']).sort_values('best_val_loss')
    chosen=selection.iloc[0]['experiment'];config=configs[chosen];epochs=int(selection.iloc[0]['best_epoch'])
    print('ALEGERE MODEL (val only):',chosen,'epoci:',epochs,flush=True)
    # TODO5: compare strength of L2 / L1 / dropout and dataset size
    reg_configs={
       'fara regularizare':dict(),
       'L2 slab (1e-4)':dict(weight_decay=1e-4),
       'L2 puternic (0.1)':dict(weight_decay=.1),
       'L1 moderat (1e-5)':dict(l1_lambda=1e-5),
       'L1 puternic (0.01)':dict(l1_lambda=.01),
       'Dropout 0.2':dict(dropout=.2),
       'Dropout 0.8':dict(dropout=.8),
    }
    for n in (500,5000):
        rng=np.random.default_rng(42); ix=rng.choice(len(ytr),size=n,replace=False)
        for name,cfg in reg_configs.items():
            _,met,h=fit_mlp(x8tr[ix],ytr[ix],x8va,yva,hidden=(128,128),epochs=45,batch_size=128,**cfg)
            register('regularization',f'{name} / N={n}',dict(n_train=n,**met),h)
    # TODO6 benchmark CPU and CUDA when available
    if run_benchmark:
        for dev in ([torch.device('cpu'),torch.device('cuda')] if torch.cuda.is_available() else [torch.device('cpu')]):
            for hidden in [(64,32),(256,256,256)]:
                for bs in (128,1024):
                    _,met,_=fit_mlp(x8tr[:8000],ytr[:8000],x8va[:4000],yva[:4000],hidden=hidden,epochs=4,batch_size=bs,device=dev)
                    rows.setdefault('benchmark',[]).append(dict(device=str(dev),hidden=str(hidden),batch_size=bs,seconds=met['seconds'],val_loss=met['best_val_loss']))
                    print('BENCHMARK',str(dev),hidden,'batch',bs,f'{met["seconds"]:.3f} sec',flush=True)
    # final test once after validation selection
    final_ids=np.concatenate([tr,va]); xfinal,yfinal=xy(cc,final_ids,FEATURES)
    x_test,y_test=xy(cc,te,FEATURES)
    final_scaler=StandardScaler().fit(xfinal)
    xfinal_s=final_scaler.transform(xfinal).astype(np.float32)
    x_test_s=final_scaler.transform(x_test).astype(np.float32)
    # independent train for selected number of epochs; no test passed to training
    seed_all()
    net=MLP(len(FEATURES),config.get('hidden',(64,32)),config.get('dropout',0.),config.get('activation','ReLU')).to(DEVICE)
    opts={'Adam':lambda:torch.optim.Adam(net.parameters(),lr=config.get('lr',.001),weight_decay=config.get('weight_decay',0.)),
          'SGD':lambda:torch.optim.SGD(net.parameters(),lr=config.get('lr',.001),momentum=.9,weight_decay=config.get('weight_decay',0.)),
          'RMSprop':lambda:torch.optim.RMSprop(net.parameters(),lr=config.get('lr',.001),weight_decay=config.get('weight_decay',0.))}
    opt=opts[config.get('optimizer','Adam')]()
    crit=nn.BCEWithLogitsLoss()
    xf=torch.as_tensor(xfinal_s,dtype=torch.float32,device=DEVICE);yf=torch.as_tensor(yfinal,dtype=torch.float32,device=DEVICE)
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(yf),device=DEVICE).split(BATCH_SIZE):
            opt.zero_grad(set_to_none=True);loss=crit(net(xf[ix]),yf[ix]);loss.backward();opt.step()
    net.eval()
    with torch.no_grad():
        logit=net(torch.as_tensor(x_test_s,dtype=torch.float32,device=DEVICE))
        probs=torch.sigmoid(logit).cpu().numpy()
    met=calc_metrics(y_test,probs)
    met.update(n_test=int(len(y_test)),log_loss=float(log_loss(y_test,probs)),selected=chosen,epochs=epochs)
    rows['test']=met; rows['test_cm']=confusion_matrix(y_test,probs>=.5,labels=[0,1]).tolist()
    print('TEST FINAL:',json.dumps(met,ensure_ascii=False),flush=True)
    # store all summaries + compact curve previews
    rows['metadata']={'data_all_labels':len(full),'data_cc_8':len(cc),'tr':len(tr),'va':len(va),'te':len(te),'positive_rate':float(ytr.mean()),'device':str(DEVICE),'seed':RANDOM_SEED}
    rows['histories']={group:{k:{x:[float(v) for v in y] for x,y in hist.items()} for k,hist in h.items()} for group,h in hists.items()}
    with open(out/'results.json','w',encoding='utf8') as f:json.dump(rows,f,ensure_ascii=False,indent=2)
    for group in ['features','missing','imbalance','selection','regularization','benchmark']:
        if group in rows:
            pd.DataFrame(rows[group]).to_csv(out/f'{group}.csv',index=False)
    return rows

if __name__=='__main__':
    import sys
    path=sys.argv[1] if len(sys.argv)>1 else 'weatherAUS.csv'
    out=sys.argv[2] if len(sys.argv)>2 else 'lab3_results'
    do_all(path,out)
