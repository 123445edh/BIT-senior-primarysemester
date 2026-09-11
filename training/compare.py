"""比较实验的验证指标、参数量和batch=1耗时；不使用测试集挑架构。"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def compare(paths, out):
    reports = [json.loads((p/'metrics.json').read_text(encoding='utf-8')) for p in paths]
    # 比较参数差异时，应保持数据、种子、训练预算一致。
    reference = reports[0]
    for r in reports[1:]:
        for key in ['manifest_sha256', 'seed', 'unique_samples']:
            if r[key] != reference[key]:
                raise ValueError(f'比较条件不一致: {key}')
        for key in ['epochs', 'batch_size', 'lr', 'weight_decay', 'label_smoothing']:
            if r['command_args'][key] != reference['command_args'][key]:
                raise ValueError(f'训练设置不一致: {key}')
        if any(r[k] != reference[k] for k in ['device', 'hardware', 'threads', 'torch']):
            raise ValueError('耗时需要相同设备/线程数/PyTorch版本，不混合比较')
    rows = []
    for path, r in zip(paths, reports):
        rows.append({'run': path.name, 'parameters': r['parameters'],
                     'checkpoint_mib': r['checkpoint_mib'],
                     'validation_accuracy': r['final']['val']['accuracy'],
                     'validation_macro_f1': r['final']['val']['macro_f1'],
                     'forward_p50_ms': r['inference_benchmark']['p50_ms'],
                     'optimization_seconds': r['optimization_seconds_total']})
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for r in rows:
        axes[0].scatter(r['parameters']/1000, r['validation_accuracy']*100)
        axes[0].annotate(r['run'], (r['parameters']/1000, r['validation_accuracy']*100), fontsize=8)
    axes[0].set(xlabel='Parameters (thousands)', ylabel='Validation accuracy (%)', title='Accuracy vs model size')
    names = [r['run'] for r in rows]
    axes[1].bar(names, [r['forward_p50_ms'] for r in rows], color='#28a78b')
    axes[1].set(title='Batch=1 forward latency', ylabel='Median milliseconds')
    axes[2].bar(names, [r['optimization_seconds'] for r in rows], color='#e99836')
    axes[2].set(title='Total optimization time', ylabel='Seconds')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    compare(args.runs, args.out)
