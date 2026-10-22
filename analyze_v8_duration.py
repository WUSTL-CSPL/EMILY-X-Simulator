#!/usr/bin/env python3
"""Paired duration effects and localization baselines; no truth-derived features."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.cache/duration_ml'))
import json
import sqlite3
import zlib
import warnings
import time
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import generate_multireceiver_dataset_v8 as g
from compare_v8_duration import VARIANTS
base=g.base
OUT=ROOT/'datasets/v8_duration_comparison'
SEEDS=[11,22,33,44,55]


def load_table(path):return [json.loads(x) for x in path.read_text().splitlines()]


def features(row):
    values=[];powers=[]
    for group in row['receivers']:
        incidents=group['incidents'];rss=[]
        for low,high in ((200e6,1200e6),(1200e6,2000e6),(2000e6,2501e6)):
            matches=[i for i in incidents if low<=i['f_center_hz']<high]
            if not matches:values.extend([0,-160,0,0,0,0,0,0]);rss.append(-160.);continue
            best=max(matches,key=lambda i:i['intensity_value']);d=best['n_time_samples']
            dbm=10*np.log10(best['intensity_value']);rss.append(dbm)
            values.extend([1,dbm,best['bw_hz']/1e6,best['n_time_pixels']/d,
                best['n_freq_pixels'],best['n_pixels']/(d*max(best['n_freq_pixels'],1)),
                int(best['ongoing']),np.log1p(len(matches))])
        powers.append(rss[0])
    # Relative received powers can remove much of the unknown common TX level.
    values.extend(powers[i]-powers[j] for i in range(len(powers)) for j in range(i))
    return np.array(values,float)


def strongest(group):
    return max(group['incidents'],key=lambda x:x['intensity_value']) if group['incidents'] else None


def metrics(errors):
    return dict(mean_km=float(np.mean(errors)),median_km=float(np.median(errors)),p90_km=float(np.quantile(errors,.9)),
                within_1km=float(np.mean(errors<=1)),within_3km=float(np.mean(errors<=3)))


def bootstrap_blocks(delta,blocks):
    rng=np.random.default_rng(20261014);names=sorted(set(blocks));indices={b:np.flatnonzero(np.array(blocks)==b) for b in names}
    means=[]
    for _ in range(3000):
        chosen=np.concatenate([indices[names[i]] for i in rng.integers(0,len(names),len(names))])
        means.append(float(np.mean(delta[chosen])))
    return dict(mean_change_km=float(np.mean(delta)),ci95_km=list(map(float,np.quantile(means,[.025,.975]))),
                geographic_test_blocks=len(names),method='Paired block bootstrap of mean error change; model seeds averaged within location.')


def main():
    report=OUT/'analysis';report.mkdir(exist_ok=False)
    selected=g.load(OUT/'sample_ids.json');proposal_ids=g.load(OUT/'proposal_ids.json')
    config=g.load(OUT/'native_60_4rx_1000/config.json')['simulation']
    forward,_=base.projection(config['profile'])
    datasets={};Xs={};Ys={};tables={};blocks={}
    for n in (3,4):
        for variant in VARIANTS:
            key=f'{variant}_{n}rx';root=OUT/f'{key}_1000';tables[key]={};Xs[key]={};Ys[key]={}
            for split in g.SPLITS:
                inp=load_table(root/split/'inputs.jsonl');labels=load_table(root/split/'labels.jsonl');truth=load_table(root/split/'ground_truth.jsonl')
                assert [r['sample_id'] for r in inp]==selected[split]==[r['sample_id'] for r in labels]==[r['sample_id'] for r in truth]
                tables[key][split]=inp;Xs[key][split]=np.array([features(r) for r in inp])
                Ys[key][split]=np.array([forward.transform(r['longitude_deg'],r['latitude_deg']) for r in labels])/1000
                blocks[split]=[r['block_id'] for r in truth]
    y=Ys['native_60_4rx'];distances={}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(selected[a])&set(selected[b]);assert not set(blocks[a])&set(blocks[b])
        distances[f'{a}_to_{b}_m']=float(cKDTree(y[a]).query(y[b])[0].min()*1000)
        assert distances[f'{a}_to_{b}_m']>=500-1e-5
    db=sqlite3.connect('file:'+str(ROOT/'runs/v8_duration_comparison/bundles.sqlite')+'?mode=ro',uri=True)
    detection={};changes={};timing=[]
    for split,ids in proposal_ids.items():
        counts={v:{'3rx':0,'4rx':0,'receiver_incidents':[0]*4,'incident_records':0} for v in VARIANTS}
        for sid in ids:
            bundle=json.loads(zlib.decompress(db.execute('select payload from bundles where id=?',(sid,)).fetchone()[0]))
            for v in VARIANTS:
                row=bundle['rows'][v];counts[v]['3rx']+=g.all_detected(row,3);counts[v]['4rx']+=g.all_detected(row,4)
                for i,r in enumerate(row['input']['receivers']):counts[v]['receiver_incidents'][i]+=bool(r['incidents']);counts[v]['incident_records']+=len(r['incidents'])
            timing.append(bundle['timing'])
        detection[split]=dict(proposals=len(ids),counts=counts)
    db.close()
    for n in (3,4):
        ref=[r for split in g.SPLITS for r in tables[f'native_60_{n}rx'][split]]
        for variant in VARIANTS:
            observed=[r for split in g.SPLITS for r in tables[f'{variant}_{n}rx'][split]]
            rss=[];bw=[];count_change=0;ongoing=[];records=0
            for a,b in zip(observed,ref):
                for ga,gb in zip(a['receivers'],b['receivers']):
                    ia,ib=strongest(ga),strongest(gb)
                    rss.append(10*np.log10(ia['intensity_value']/ib['intensity_value']))
                    bw.append((ia['bw_hz']-ib['bw_hz'])/1e6)
                    count_change+=len(ga['incidents'])!=len(gb['incidents'])
                    ongoing.extend(i['ongoing'] for i in ga['incidents']);records+=len(ga['incidents'])
            changes[f'{variant}_{n}rx']=dict(median_abs_strongest_intensity_change_db=float(np.median(np.abs(rss))),
                p90_abs_strongest_intensity_change_db=float(np.quantile(np.abs(rss),.9)),mean_signed_intensity_change_db=float(np.mean(rss)),
                median_abs_bandwidth_change_mhz=float(np.median(np.abs(bw))),receiver_incident_count_change_fraction=count_change/(1000*n),
                total_incidents=records,ongoing_fraction=float(np.mean(ongoing)))
    bench=g.load(OUT/'benchmark.json');speed={}
    for d in (20,30,60):
        runs=[b for b in bench if b['duration_s']==d]
        components=[t[f'native_{d}'] for t in timing]
        speed[str(d)]=dict(benchmark_mean_wall_s=float(np.mean([b['wall_s'] for b in runs])),
            benchmark_repeats_wall_s=[b['wall_s'] for b in runs],benchmark_count_per_repeat=runs[0]['count'],
            seconds_per_scene=float(np.mean([b['wall_s']/b['count'] for b in runs])),
            component_median_s={k:float(np.median([c[k] for c in components])) for k in components[0]})
    for d in ('20','30','60'):speed[d]['speedup_vs_60']=speed['60']['seconds_per_scene']/speed[d]['seconds_per_scene']
    base.write_json(report/'observations.json',dict(detection=detection,paired_incident_changes=changes,timing=speed,minimum_split_distances_m=distances))
    predictions={};model_results={};hyperparameters={};convergence=[]
    with threadpool_limits(limits=1):
        for n in (3,4):
            ref=f'native_60_{n}rx';scaler=StandardScaler().fit(Xs[ref]['train']);target_scale=StandardScaler().fit(Ys[ref]['train'])
            # Tune only on 60-second validation, then freeze the choice across durations.
            val_options=[]
            for alpha in (.001,.1):
                model=MLPRegressor(hidden_layer_sizes=(64,32),alpha=alpha,learning_rate_init=.002,batch_size=64,max_iter=600,
                    n_iter_no_change=40,tol=1e-5,random_state=11,early_stopping=False)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConvergenceWarning)
                    model.fit(scaler.transform(Xs[ref]['train']),target_scale.transform(Ys[ref]['train']))
                pred=target_scale.inverse_transform(model.predict(scaler.transform(Xs[ref]['validation'])))
                val_options.append((float(np.linalg.norm(pred-Ys[ref]['validation'],axis=1).mean()),alpha))
            alpha=min(val_options)[1]
            hyperparameters[str(n)]=dict(mlp_alpha=alpha,mlp_validation_options=val_options,mlp_hidden=[64,32],mlp_max_iter=600,
                mlp_seeds=SEEDS,extra_trees_estimators=256,extra_trees_min_samples_leaf=3,
                selection='MLP alpha selected on native 60s validation only; shared across durations. Each model fits its scaler on its own training set.')
            for variant in VARIANTS:
                key=f'{variant}_{n}rx';scaler=StandardScaler().fit(Xs[key]['train']);tgt=StandardScaler().fit(Ys[key]['train'])
                xt=scaler.transform(Xs[key]['train']);xv=scaler.transform(Xs[key]['validation']);xe=scaler.transform(Xs[key]['test'])
                yt=tgt.transform(Ys[key]['train'])
                predictions[key]={};model_results[key]={}
                for family,seeds in [('MLP',SEEDS),('ExtraTrees',[11])]:
                    errors=[];val_errors=[];trained=[];train_errors=[];cross={d:[] for d in (20,30,60)}
                    for seed in seeds:
                        model=(MLPRegressor(hidden_layer_sizes=(64,32),alpha=alpha,learning_rate_init=.002,batch_size=64,
                            max_iter=600,n_iter_no_change=40,tol=1e-5,random_state=seed,early_stopping=False) if family=='MLP' else
                            ExtraTreesRegressor(n_estimators=256,min_samples_leaf=3,random_state=seed,n_jobs=1))
                        start=time.perf_counter()
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always',ConvergenceWarning);model.fit(xt,yt)
                        convergence.extend(dict(dataset=key,model=family,seed=seed,message=str(w.message)) for w in caught)
                        pred=tgt.inverse_transform(model.predict(xe));err=np.linalg.norm(pred-Ys[key]['test'],axis=1)
                        validation=np.linalg.norm(tgt.inverse_transform(model.predict(xv))-Ys[key]['validation'],axis=1)
                        train=np.linalg.norm(tgt.inverse_transform(model.predict(xt))-Ys[key]['train'],axis=1)
                        errors.append(err);val_errors.append(validation);train_errors.append(train);trained.append(time.perf_counter()-start)
                        for test_d in (20,30,60):
                            other=f'native_{test_d}_{n}rx'
                            prediction=tgt.inverse_transform(model.predict(scaler.transform(Xs[other]['test'])))
                            cross[test_d].append(np.linalg.norm(prediction-Ys[other]['test'],axis=1))
                    e=np.array(errors);predictions[key][family]=e
                    model_results[key][family]=dict(test=metrics(e.mean(axis=0)),test_mean_km_per_seed=list(map(float,e.mean(axis=1))),
                        validation_mean_km=float(np.mean(val_errors)),train_mean_km=float(np.mean(train_errors)),fit_and_predict_seconds=trained,
                        transfer_to_native={str(d):metrics(np.mean(errs,axis=0)) for d,errs in cross.items()},
                        note='For MLP, metrics use mean error over training seeds at each test location, not a prediction ensemble.')
                    print(key,family,model_results[key][family]['test'],flush=True)
            # Constant-coordinate baseline, fitted on training labels only.
            constant=np.linalg.norm(Ys[ref]['test']-Ys[ref]['train'].mean(axis=0),axis=1)
            hyperparameters[str(n)]['train_centroid_test']=metrics(constant)
    differences={}
    for n in (3,4):
        for variant in VARIANTS:
            if variant=='native_60':continue
            for family in ('MLP','ExtraTrees'):
                delta=predictions[f'{variant}_{n}rx'][family].mean(axis=0)-predictions[f'native_60_{n}rx'][family].mean(axis=0)
                differences[f'{variant}_{n}rx_{family}']=bootstrap_blocks(delta,blocks['test'])
    np.savez_compressed(report/'test_errors.npz',**{f'{k}_{m}':v for k,ms in predictions.items() for m,v in ms.items()})
    base.write_json(report/'model_results.json',dict(results=model_results,paired_differences=differences,hyperparameters=hyperparameters,
        convergence_warnings=convergence,sklearn_version=sklearn.__version__,features='Three spectral groups per RX: presence, strongest dBm, bandwidth, time occupancy, frequency pixels, pixel occupancy, ongoing, log incident count; plus pairwise low-band dBm differences.',
        inputs='Only incident fields and fixed receiver ordering. Source power, path loss, source coordinates and block IDs are excluded from features. Coordinates are targets; block IDs are used only for evaluation bootstrap.',
        sample_counts={'train':700,'validation':150,'test':150},classification_tested=False))
    plot_and_write(report,speed,detection,changes,model_results,differences,hyperparameters)


def plot_and_write(report,speed,detection,changes,results,differences,hyper):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    ds=[20,30,60];bars=axes[0].bar([str(d) for d in ds],[speed[str(d)]['seconds_per_scene'] for d in ds],color=['#3884ad','#64a57a','#777b87'])
    axes[0].bar_label(bars,fmt='%.3f',padding=3);axes[0].set(xlabel='Native duration (s)',ylabel='Wall seconds / scene',title='Throughput benchmark (16 workers)',ylim=(0,max(speed[str(d)]['seconds_per_scene'] for d in ds)*1.22))
    for ax,n in zip(axes[1:],(3,4)):
        for family,color in [('MLP','#3884ad'),('ExtraTrees','#bd7437')]:
            vals=[results[f'native_{d}_{n}rx'][family]['test']['mean_km'] for d in ds]
            spreads=[float(np.std(results[f'native_{d}_{n}rx'][family]['test_mean_km_per_seed'],ddof=1)) if family=='MLP' else 0. for d in ds]
            ax.errorbar(ds,vals,yerr=spreads,fmt='o-',capsize=4,label=family+(' (training-seed SD)' if family=='MLP' else ''),color=color)
        ax.set(xlabel='Native duration (s)',ylabel='Mean test location error (km)',title=f'{n} receivers; same 150 test positions',ylim=(0,6));ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(report/'duration_summary.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,n in zip(axes,(3,4)):
        for prefix,style in [('native','o-'),('prefix','s--')]:
            values=[results[f'{prefix}_{d}_{n}rx']['MLP']['test']['mean_km'] for d in (20,30)]+[results[f'native_60_{n}rx']['MLP']['test']['mean_km']]
            ax.plot(ds,values,style,label=prefix)
        ax.set(title=f'MLP, {n} receivers',xlabel='Duration (s)',ylabel='Mean test error (km)',ylim=(0,6));ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(report/'prefix_control.png',dpi=160);plt.close(fig)
    total=sum(x['proposals'] for x in detection.values())
    lines=['# 20、30、60 秒对训练数据的影响','',
        '本次结果支持优先试用 30 秒：直接生成约快 1.70 倍，四接收器 MLP 平均误差与 60 秒版几乎相同。20 秒约快 2.19 倍，当前基线没有出现一致的大幅退化，但 MLP 训练波动更大。结论限于这批静态、持续 loaded LTE 场景。', '',
        '## 怎么比较的','',
        '使用 v8 的 20×20 km 区域、33–43 dBm 随机发射功率、每个接收器独立的 2 dB 测量偏差、原有全向接收器和 P.452/SRTM。只有观测时长不同，时间分辨率仍为 1 秒。',
        '3 和 4 个接收器各做了 20、30、60 秒版本，每版 1,000 个场景，按原地理区域分成 700/150/150。全部版本使用完全相同的位置、功率、测量偏差和 split。',
        f'先评估 {total:,} 个相同候选位置，再保留所有时长、所有对照下四个接收器都能检测到的 1,000 个位置。三接收器版本使用其中前三个接收器。因此它是共同可检测位置上的公平对照，不能代表所有位置；筛选前的检测率也单独统计。',
        '额外做了 prefix 对照：把同一条 60 秒频谱截取前 20/30 秒，再重新跑 EMILY 检测。native 是直接把模拟时长设为 20/30/60 秒。两者分开报告。',
        '原因是原代码按观测长度生成、平滑并归一化噪声/慢衰落；同一 seed 在不同长度下并不生成同一条信号的前缀。prefix 对照排除了这一混杂因素。', '',
        '## 速度','', '| 观测时长 | 每场景实际耗时，秒 | 相比 60 秒提速 |','| --- | ---: | ---: |']
    for d in ds:lines.append(f"| {d}s | {speed[str(d)]['seconds_per_scene']:.4f} | {speed[str(d)]['speedup_vs_60']:.2f}× |")
    lines+=['','使用同一组 128 个场景、16 个进程，独立测两轮，顺序交替；每次重新执行传播、频谱和检测。没有把 prefix 截取或旧数据缓存的速度当作直接生成的速度。IQ 模板初始化、写盘和 ZIP 不计入这张表；百万样本总耗时不能只按这张表精确预测。', '',
        '## 筛选前检测率','', '| 方式 | 三接收器全部检测到 | 四接收器全部检测到 |','| --- | ---: | ---: |']
    for v in VARIANTS:
        a=sum(x['counts'][v]['3rx'] for x in detection.values());b=sum(x['counts'][v]['4rx'] for x in detection.values())
        lines.append(f'| {v} | {a/total:.2%} | {b/total:.2%} |')
    lines+=['','## Incident 变化（共同可检测位置）','', '| 方式，4 RX | 最强 incident 强度变化绝对值中位数，dB | 90% 分位，dB | incident 数量变化的接收器比例 |','| --- | ---: | ---: | ---: |']
    for v in VARIANTS:
        r=changes[f'{v}_4rx'];lines.append(f"| {v} | {r['median_abs_strongest_intensity_change_db']:.3f} | {r['p90_abs_strongest_intensity_change_db']:.3f} | {r['receiver_incident_count_change_fraction']:.2%} |")
    lines+=['','变化以同位置 60 秒为基准。最强 incident 对比不保证跨版本属于完全相同的分割事件；详细频宽、事件总数和截尾比例见 observations.json。',
        '当前模拟每秒一行，所以 20/30/60 行对应的 incident observation_duration_sec 分别是 19/29/59 秒（首尾时间戳之差），并非少模拟了一秒。ongoing incident 的结束时间和 duration 保留 null。', '',
        '## 实际训练结果','',
        '用同一组数据训练一个两层 MLP（64、32 个隐藏单元，五个训练 seed）和一个 ExtraTrees 基线（256 棵树）。预测本地平面坐标，计算到真实位置的公里误差。特征只来自 incident；功率真值、传播损耗和发射机坐标不作为输入。',
        'MLP 的正则参数只用 60 秒版本的 validation 选择，随后在所有时长保持一致。标准化只拟合各版本训练集。时间像素数转换为占观测窗口的比例，避免把更短窗口的像素数机械地当作不同信号。', '',
        '| RX | 时长 | MLP 平均 / 中位 / P90 误差，km | ExtraTrees 平均误差，km |','| --- | --- | --- | ---: |']
    for n in (3,4):
        for d in ds:
            r=results[f'native_{d}_{n}rx'];m=r['MLP']['test'];lines.append(f"| {n} | {d}s | {m['mean_km']:.3f} / {m['median_km']:.3f} / {m['p90_km']:.3f} | {r['ExtraTrees']['test']['mean_km']:.3f} |")
    lines+=['','MLP 表中先对每个测试位置的五次训练误差取平均，再统计位置分布；不是五模型预测集成。每个 seed 的结果、训练/验证误差和收敛提示见 model_results.json。', '',
        '| RX | 对比 | MLP 平均误差变化，km | 95% block-bootstrap 区间，km |','| --- | --- | ---: | --- |']
    for n in (3,4):
        for v in ('native_20','native_30','prefix_20','prefix_30'):
            x=differences[f'{v}_{n}rx_MLP'];lo,hi=x['ci95_km'];lines.append(f"| {n} | {v} − native_60 | {x['mean_change_km']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['','正数表示比 60 秒差，负数表示更好。区间按测试地理块成组重采样，保留空间相关性；但只有四个测试地理块和 150 个测试位置，不能把区间包含零解释为“已证明等效”。', '', '## 用短窗口训练，测试仍用 60 秒','',
        '下面保持同一批 60 秒测试观测，只改变训练数据的时长。标准化和模型仍只从对应训练集拟合。', '',
        '| RX | 训练时长 | 测试时长 | MLP 平均误差，km | ExtraTrees 平均误差，km |',
        '| --- | --- | --- | ---: | ---: |']
    for n in (3,4):
        for d in ds:
            r=results[f'native_{d}_{n}rx'];lines.append(f"| {n} | {d}s | 60s | {r['MLP']['transfer_to_native']['60']['mean_km']:.3f} | {r['ExtraTrees']['transfer_to_native']['60']['mean_km']:.3f} |")
    lines+=['','50 次最终 MLP 训练中，3 次到 600 轮仍触发未收敛提示（20 秒 native 两次、20 秒 prefix 一次）。这里比较的是相同训练预算下的基线，微小差异不宜过度解释。图中误差条仅表示训练 seed 之间的标准差，不是地理泛化置信区间。',
        '本实验没有测试用户已有模型。它只检验这两个基线在固定 LTE loaded、静止发射机、固定接收器和这一区域上的表现。对短暂发射、负载切换、更多发射机类型或动态源，短窗口可能损失本实验没有覆盖的信息。', '',
        '![速度与定位误差](duration_summary.png)', '', '![相同信号前缀对照](prefix_control.png)', '',
        '## 数据与复现','',
        '六个 native ZIP 在上一层目录；prefix 控制数据放在对应文件夹中。所有版本共享位置和 split，不能拼接为互相独立的数据。原 v8 文件没有覆盖。',
        '```bash','../.venv/bin/python -m pip install --target .cache/duration_ml --no-deps -r configs/duration_ml_requirements.txt','../.venv/bin/python compare_v8_duration.py','../.venv/bin/python analyze_v8_duration.py','```',
        '生成中断时加 --resume。分析报告目录已存在时拒绝覆盖。ML 依赖安装在 .cache/duration_ml，未更新原模拟器环境；本次使用 scikit-learn '+sklearn.__version__+'。','']
    (report/'README_zh.md').write_text('\n'.join(lines))
    base.write_json(report/'verification.json',dict(passed=True,aligned_labels_and_ids=True,all_1000_positions_shared=True,
        split_geographic_separation_m=g.load(report/'observations.json')['minimum_split_distances_m'],
        no_truth_features=True,scalers_fit_train_only=True,test_not_used_for_hyperparameter_selection=True))

if __name__=='__main__':main()
