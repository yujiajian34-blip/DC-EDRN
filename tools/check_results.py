"""Recompute selected paper aggregates from archived numerical files; no ML imports."""
from pathlib import Path
import csv
import gzip
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]

def rows(name):
    op = gzip.open if name.endswith('.gz') else open
    with op(ROOT/name,'rt',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    checks = []
    def check(name,ok): checks.append({'check':name,'passed':bool(ok)})
    data = rows('results/primary_three_seed/metrics_by_seed.csv')
    values = [float(r['F1']) for r in data if r['Model']=='DC_EDRN']
    check('DC-EDRN F1 mean/seed SD round to 0.932/0.008',len(values)==3 and round(statistics.mean(values),3)==.932 and round(statistics.stdev(values),3)==.008)
    check('Primary denominator is 885',all(float(r['N'])==885 for r in data))
    predictions = rows('results/primary_three_seed/predictions.csv.gz')
    groups = {}
    for r in predictions: groups.setdefault((r['model'],int(float(r['seed']))),[]).append(r)
    for (model,seed),g in groups.items():
        tp = sum(int(r['y'])==1 and int(r['prediction'])==1 for r in g)
        fp = sum(int(r['y'])==0 and int(r['prediction'])==1 for r in g)
        fn = sum(int(r['y'])==1 and int(r['prediction'])==0 for r in g)
        f1 = 2*tp/(2*tp+fp+fn)
        expected = [r for r in data if r['Model']==model and int(r['Seed'])==seed]
        check(f'{model}:{seed} row predictions match reported F1',len(g)==885 and len({r['id'] for r in g})==885 and len(expected)==1 and math.isclose(f1,float(expected[0]['F1']),abs_tol=1e-12))
    perf = rows('results/performance_3090/screening_cost.csv')
    for model,batch,expected in [('DC_EDRN_original',1,66.85),('DC_EDRN_optimized',1,48.61),('DC_EDRN_original',32,69.26),('DC_EDRN_optimized',32,50.96)]:
        g = [r for r in perf if r['Model']==model and int(r['Batch'])==batch and r['Measurement']=='tokenize_transfer_forward_scores']
        check(f'Table 6 {model} batch {batch}',len(g)==1 and round(float(g[0]['Mean_ms']),2)==expected)
    parity = json.loads((ROOT/'results/performance_3090/parity.json').read_text())
    check('3,827-row parity and unchanged decisions',parity['status']=='PASS' and parity['evaluated_rows']==3827 and all(v==0 for v in parity['prediction_differences'].values()))
    live = rows('results/live_gateway_timerfix/summary.csv')
    check('48 completed worker conditions',len(live)==48)
    for rate,policy,throughput,p95 in [(10,'immediate',10.0,55.3),(10,'dynamic32_5ms',10.0,61.1),(40,'immediate',19.6,19854.3),(40,'dynamic32_5ms',39.9,100.3)]:
        g = [r for r in live if r['pattern']=='periodic' and float(r['target_rate_texts_s'])==rate and r['policy']==policy]
        check(f'Worker periodic {rate}/s {policy}',len(g)==3 and round(statistics.mean(float(r['achieved_throughput_texts_s']) for r in g),1)==throughput and round(statistics.mean(float(r['p95_e2e_ms']) for r in g),1)==p95)
    result = {'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL',
              'scope':'Archived result arithmetic, not fresh inference or GPU execution','checks':checks}
    print(json.dumps(result,indent=2))
    raise SystemExit(result['status'] != 'PASS')

if __name__ == '__main__': main()
