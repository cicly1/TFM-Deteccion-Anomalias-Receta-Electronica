"""Validación externa agrupada por paciente y ajuste en validación interna.
Las etiquetas de la validación interna se usan SOLO para elegir hiperparámetros;
las etiquetas externas se usan SOLO para evaluación. Experimento con datos sintéticos.
"""
import argparse,time, itertools
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support
SEEDS=[11,22,33,44,55]
DROP=['id_registro','id_paciente_sintetico','fecha_prescripcion','fecha_dispensacion','medicamento','etiqueta_anomalia','tipo_anomalia','duplicidad_observada']
GRIDS={
 'Isolation Forest':[dict(n_estimators=n,max_samples=s,max_features=f) for n,s,f in itertools.product([100,150,250,400],[.65,1.0],[.7,1.0])],
 'LOF':[dict(n_neighbors=n) for n in [20,35,50]],
 'One-Class SVM':[dict(nu=n,gamma=g) for n,g in itertools.product([.05,.10],[.01,.1,'scale'])]
}
def model(name,params,seed):
 if name=='Isolation Forest':return IsolationForest(contamination='auto',random_state=seed,n_jobs=1,**params)
 if name=='LOF':return LocalOutlierFactor(novelty=True,contamination='auto',n_jobs=1,**params)
 return OneClassSVM(kernel='rbf',**params)
def preprocess(x,fit_idx,other_idx,context=False):
 a=x.iloc[fit_idx].copy();b=x.iloc[other_idx].copy()
 if context:
  pairs=a['especialidad'].astype(str)+'|'+a['grupo_atc'].astype(str)
  freq=pairs.value_counts(normalize=True)
  for frame in (a,b):
   pp=frame['especialidad'].astype(str)+'|'+frame['grupo_atc'].astype(str)
   frame['frecuencia_par_especialidad_atc']=pp.map(freq).fillna(0).to_numpy()
 cats=list(a.select_dtypes(include=['object']).columns)
 nums=[c for c in a if c not in cats]
 prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),('num',StandardScaler(),nums)])
 return prep.fit_transform(a),prep.transform(b)
def main(path, output_dir):
 output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(path);y=df.etiqueta_anomalia.to_numpy();groups=df.id_paciente_sintetico.to_numpy();X=df.drop(columns=DROP)
 records=[];typerecs=[];sensitivity=[]
 for seed in SEEDS:
  outer_tr,outer_te=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed).split(X,y,groups))
  inner_rel,valid_rel=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+100).split(X.iloc[outer_tr],y[outer_tr],groups[outer_tr]))
  inner_tr=outer_tr[inner_rel];inner_va=outer_tr[valid_rel]
  for scenario in ['solo_normales','contaminado']:
   fit_inner=inner_tr[y[inner_tr]==0] if scenario=='solo_normales' else inner_tr
   fit_outer=outer_tr[y[outer_tr]==0] if scenario=='solo_normales' else outer_tr
   for context in [False,True]:
    A,V=preprocess(X,fit_inner,inner_va,context)
    B,T=preprocess(X,fit_outer,outer_te,context)
    for name,grid in GRIDS.items():
     best=None
     for params in grid:
      m=model(name,params,seed);m.fit(A)
      score=-m.decision_function(V)
      metric=average_precision_score(y[inner_va],score)
      if best is None or metric>best[0]:best=(metric,params)
     t0=time.perf_counter();m=model(name,best[1],seed);m.fit(B);train_s=-m.decision_function(B);test_s=-m.decision_function(T);seconds=time.perf_counter()-t0
     # Umbral de entrenamiento predefinido: percentil 95 fijo de puntuaciones de entrenamiento, sin etiquetas externas.
     threshold=np.quantile(train_s,0.95)
     pred=(test_s>=threshold).astype(int)
     yt=y[outer_te];p,r,f,_=precision_recall_fscore_support(yt,pred,average='binary',zero_division=0)
     npos=int(yt.sum());idx=np.argsort(-test_s);k=min(50,len(idx));rk=min(npos,len(idx))
     row=dict(semilla=seed,escenario=scenario,contexto_atc=context,modelo=name,n_entrenamiento=len(fit_outer),n_prueba=len(outer_te),anomalias_prueba=npos,precision=p,recall=r,f1=f,roc_auc=roc_auc_score(yt,test_s),pr_auc=average_precision_score(yt,test_s),precision_50=float(yt[idx[:k]].mean()),recall_50=float(yt[idx[:k]].sum()/npos),r_precision=float(yt[idx[:rk]].mean()),tiempo_ajuste_puntuacion_s=seconds,pr_auc_validacion_interna=best[0],hiperparametros=str(best[1]))
     records.append(row)
     for percentile in (90,95,97):
      cutoff=np.quantile(train_s,percentile/100)
      alerts=(test_s>=cutoff).astype(int)
      pp,rr,ff,_=precision_recall_fscore_support(yt,alerts,average='binary',zero_division=0)
      sensitivity.append(dict(semilla=seed,escenario=scenario,contexto_atc=context,modelo=name,percentil=percentile,umbral=cutoff,precision=pp,recall=rr,f1=ff,alertas=int(alerts.sum()),registros_prueba=len(yt),anomalias_prueba=npos,fraccion_alertas=float(alerts.mean())))
     for kind in sorted(df.tipo_anomalia.unique()):
      if kind=='normal':continue
      mask=df.tipo_anomalia.iloc[outer_te].to_numpy()==kind
      typerecs.append(dict(semilla=seed,escenario=scenario,contexto_atc=context,modelo=name,tipo=kind,n=int(mask.sum()),detectados=int(pred[mask].sum())))
     print(seed,scenario,context,name,round(row['pr_auc'],3),best[1],flush=True)
 pd.DataFrame(records).to_csv(output_dir/'resultados_validacion_definitiva.csv',index=False)
 pd.DataFrame(typerecs).to_csv(output_dir/'resultados_tipologias_definitiva.csv',index=False)
 pd.DataFrame(sensitivity).to_csv(output_dir/'resultados_sensibilidad_p90_p95_p97.csv',index=False)
 assert (len(records),len(typerecs),len(sensitivity))==(60,360,180)
 print('DONE',len(records),len(typerecs),len(sensitivity),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--input',default=Path(__file__).resolve().parents[1]/'datos'/'dataset_tfm_definitivo.csv');parser.add_argument('--output-dir',default=Path(__file__).resolve().parents[1]/'resultados_reproducidos');args=parser.parse_args();main(args.input,args.output_dir)
