"""Generador NUEVO, completamente sintético, con duplicidades longitudinales verificables.
No lee ningún fichero histórico. Python 3.9+ y pandas para exportar CSV.
Uso: python generador_tfm_definitivo.py --seed 20261117 --output dataset_tfm_definitivo.csv
El código se ha desarrollado con asistencia de ChatGPT; requiere revisión científica.
"""
import argparse, random, hashlib, json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

CATALOG = [
 ('A10BA','Metformina','Endocrinología',500,2000),
 ('C09AA','Enalapril','Cardiología',5,20),
 ('C10AA','Atorvastatina','Cardiología',10,40),
 ('N02BE','Paracetamol','Medicina familiar',500,3000),
 ('J01CA','Amoxicilina','Medicina familiar',500,3000),
 ('R03AC','Salbutamol','Neumología',100,800),
 ('N06AB','Sertralina','Psiquiatría',50,200),
 ('M01AE','Ibuprofeno','Traumatología',400,1800),
]
TYPES={'dosis_extrema':44,'frecuencia_alta':39,'duracion_excesiva':33,'duplicidad_temporal':41,'especialidad_atipica':35,'dispensacion_atipica':48}
SPECIALTIES=sorted(set(x[2] for x in CATALOG))

def generate(seed=20261117):
 r=random.Random(seed)
 rows=[]
 # Primero se generan 2400 prescripciones normales; pacientes recurrentes
 # con fechas separadas y un catálogo farmacológico coherente.
 for i in range(2400):
  atc,med,spec,unit,maximum=r.choice(CATALOG)
  patient=r.randint(1,900)
  start=date(2025,1,1)+timedelta(days=r.randint(0,364))
  # Evita solapamientos normales del mismo ATC y paciente.
  for attempt in range(200):
   if not any(x["id_paciente_sintetico"]==f"P{patient:04d}" and x["grupo_atc"]==atc and abs((x["fecha_prescripcion"]-start).days)<=90 for x in rows): break
   patient=r.randint(1,900)
  frequency=r.choice([1,1,1,2,2,3]); dose=unit*frequency
  delay=r.choices([0,1,2,3,5,7,10],[10,20,20,20,12,10,8])[0]
  rows.append(dict(id_registro=f'RX{i+1:05d}',id_paciente_sintetico=f'P{patient:04d}',edad=r.randint(18,92),sexo=r.choice(['F','M']),ambito=r.choices(['AP','AE'],[.78,.22])[0],centro=f'C{r.randint(1,35):03d}',especialidad=spec,grupo_atc=atc,medicamento=med,dosis_mg_dia=dose,frecuencia_dia=frequency,duracion_dias=r.choice([7,14,28,30,60,90]),fecha_prescripcion=start,fecha_dispensacion=start+timedelta(days=delay),retraso_dispensacion_dias=delay,etiqueta_anomalia=0,tipo_anomalia='normal'))
 # Las anomalías son registros nuevos, no reetiquetados, y cada clase se
 # define mediante una regla explícita y comprobable.
 for kind,n in TYPES.items():
  for _ in range(n):
   base=r.choice(rows[:2400]).copy(); row=base.copy()
   row['id_registro']=f'RX{len(rows)+1:05d}'
   row['etiqueta_anomalia']=1; row['tipo_anomalia']=kind
   if kind!='duplicidad_temporal': row['id_paciente_sintetico']=f'P{10000+len(rows):05d}'
   if kind=='dosis_extrema':
    maxdose=next(x[4] for x in CATALOG if x[0]==row['grupo_atc'])
    row['dosis_mg_dia']=maxdose*r.choice([3,4,5])
   elif kind=='frecuencia_alta':
    row['frecuencia_dia']=r.randint(7,12)
    unit=next(x[3] for x in CATALOG if x[0]==row['grupo_atc'])
    row['dosis_mg_dia']=unit*row['frecuencia_dia']
   elif kind=='duracion_excesiva': row['duracion_dias']=r.randint(450,900)
   elif kind=='duplicidad_temporal':
    # Copia real de un tratamiento anterior del mismo paciente, mismo ATC,
    # con fecha posterior 0-2 días y solapamiento con tratamiento original.
    row['fecha_prescripcion']=base['fecha_prescripcion']+timedelta(days=r.randint(0,2))
    row['fecha_dispensacion']=row['fecha_prescripcion']+timedelta(days=base['retraso_dispensacion_dias'])
   elif kind=='especialidad_atipica':
    row['especialidad']=r.choice([s for s in SPECIALTIES if s!=base['especialidad']])
   elif kind=='dispensacion_atipica':
    row['retraso_dispensacion_dias']=r.randint(75,180)
    row['fecha_dispensacion']=row['fecha_prescripcion']+timedelta(days=row['retraso_dispensacion_dias'])
   rows.append(row)
 # Construcción cronológica de variables históricas SOLO a partir de eventos
 # previos. La duplicidad se calcula, no se suministra como indicador.
 rows.sort(key=lambda x:(x['fecha_prescripcion'],x['id_registro']))
 history={}
 for row in rows:
  date0=row['fecha_prescripcion']; key=row['id_paciente_sintetico']
  prior=history.get(key,[])
  row['prescripciones_previas_30d']=sum(0<=(date0-x['fecha_prescripcion']).days<=30 for x in prior)
  row['dias_desde_ultima_prescripcion']=(date0-max(x['fecha_prescripcion'] for x in prior)).days if prior else 365
  row['duplicidad_observada']=int(any(x['grupo_atc']==row['grupo_atc'] and 0<=(date0-x['fecha_prescripcion']).days<=2 and date0<x['fecha_prescripcion']+timedelta(days=x['duracion_dias']) for x in prior))
  row['mes']=date0.month;row['dia_semana']=date0.weekday()
  history.setdefault(key,[]).append(row)
 for i,row in enumerate(rows,1):
  row['id_registro']=f'RX{i:05d}'
  row['fecha_prescripcion']=row['fecha_prescripcion'].isoformat()
  row['fecha_dispensacion']=row['fecha_dispensacion'].isoformat()
 return pd.DataFrame(rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261117);p.add_argument('--output',default='dataset_tfm_definitivo.csv');a=p.parse_args()
 df=generate(a.seed); path=Path(a.output);df.to_csv(path,index=False,encoding='utf-8-sig')
 counts=df.tipo_anomalia.value_counts().to_dict()
 meta=dict(seed=a.seed,total=len(df),counts=counts,columns=list(df.columns),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),provenance='Generación desde cero; no utiliza dataset previo',ai='Código desarrollado con apoyo de ChatGPT (OpenAI)',limitations='Reglas sintéticas, sin validación clínica; no utilizar etiquetas ni duplicidad_observada como predictores')
 path.with_suffix('.metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 print(counts,'duplicidades observadas',int(df.duplicidad_observada.sum()))
if __name__=='__main__':main()
