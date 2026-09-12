"""Swin训练、准确率/耗时/数据量记录。测试集不参与每轮选择。
示例: python train.py --data-dir ../data/final/data --out runs/base --epochs 30
轻量化试验添加 --skip-test；选定架构后再做最终测试。
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import FAMILIES, PREPROCESS, ImageDataset, read_manifest
from models import SwinClassifier
from plots import dataset_plot, history_plot, final_plot


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def choose_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA不可用，请检查PyTorch安装或使用--device cpu')
    return torch.device('cuda' if name == 'cuda' or name == 'auto' and torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


@torch.inference_mode()
def evaluate(model, loader, device):
    """统一使用无权重交叉熵。耗时包括加载、传输、前向、指标计算。"""
    model.eval()
    sync(device); start = time.perf_counter()
    total_loss, count = 0.0, 0
    cm = torch.zeros((9, 9), dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise ValueError('模型输出含NaN/Inf')
        total_loss += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        predicted = logits.argmax(1)
        cm += torch.bincount(labels * 9 + predicted, minlength=81).reshape(9, 9).cpu()
        count += labels.numel()
    sync(device); seconds = time.perf_counter() - start
    if not count:
        raise ValueError('评估集为空')
    matrix = cm.numpy()
    tp = np.diag(matrix)
    denom = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(9), where=denom != 0)
    return {'samples': count, 'accuracy': float(tp.sum()/count), 'macro_f1': float(f1.mean()),
            'loss': total_loss/count, 'seconds': seconds, 'samples_per_second': count/seconds,
            'confusion_matrix': matrix.tolist(), 'family_order': FAMILIES}


@torch.inference_mode()
def benchmark(model, device, repeats=30):
    """仅batch=1模型前向；预热后计时，排除读文件、预处理和CPU/GPU传输。"""
    model.eval()
    x = torch.zeros(1, 1, 32, 32, device=device)
    for _ in range(5):
        model(x)
    times = []
    for _ in range(repeats):
        sync(device); start = time.perf_counter()
        model(x)
        sync(device); times.append((time.perf_counter()-start)*1000)
    return {'batch_size': 1, 'warmup': 5, 'repeats': repeats,
            'mean_ms': float(np.mean(times)), 'p50_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)), 'scope': 'forward_only_on_device'}


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError('epochs/batch-size/threads必须为正，workers不能为负')
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError('无效优化器或label-smoothing参数')
    config = {'num_classes': 9, 'embed_dim': args.embed_dim, 'depths': args.depths,
              'num_heads': args.heads, 'mlp_ratio': args.mlp_ratio,
              'dropout': args.dropout, 'drop_path': args.drop_path}
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    model = SwinClassifier(**config).to(device)
    # 不覆盖原有实验记录。
    args.out.mkdir(parents=True, exist_ok=False)
    start_all = time.perf_counter()
    frame = read_manifest(args.data_dir)
    frame.drop(columns='path').to_csv(args.out/'manifest.csv', index=False)
    dataset_plot(frame, args.out)
    split_frames = {s: frame.loc[frame.split.eq(s)] for s in ['train', 'val', 'test']}
    datasets = {s: ImageDataset(f) for s, f in split_frames.items()}
    common = {'batch_size': args.batch_size, 'num_workers': args.workers,
              'pin_memory': device.type == 'cuda', 'worker_init_fn': seed_worker}
    # 独立生成器让评估不会改变下一轮训练的打乱顺序。
    train_loader = DataLoader(datasets['train'], shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), **common)
    loaders = {s: DataLoader(d, shuffle=False,
                 generator=torch.Generator().manual_seed(args.seed+1), **common) for s, d in datasets.items()}
    counts = np.bincount(split_frames['train'].Class.to_numpy(dtype=int)-1, minlength=9)
    weights = torch.tensor(counts.sum()/(9*counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {'architecture': 'repository SwinClassifier', 'source_commit': '92a2f055fa96b1c2e3af2383eebaf38a1235c232',
                'config': config, 'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'unique_samples': {s: len(d) for s, d in datasets.items()},
                'seed': args.seed, 'device': str(device), 'python': platform.python_version(),
                'torch': str(torch.__version__), 'numpy': np.__version__,
                'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or platform.machine(),
                'threads': args.threads, 'command_args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'manifest_sha256': hashlib.sha256((args.data_dir/'dataset.csv').read_bytes()).hexdigest(),
                'class_weights': weights.cpu().tolist(), 'preprocess': PREPROCESS}
    write_json(args.out/'config.json', metadata)
    print(f"Swin参数量 {metadata['parameters']:,}; 数据量 {metadata['unique_samples']}; 设备 {device}", flush=True)
    history, best_f1, best_epoch = [], -1.0, None
    for epoch in range(1, args.epochs+1):
        model.train(); sync(device); start = time.perf_counter()
        samples_seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise ValueError('训练loss含NaN/Inf')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            samples_seen += labels.numel()
        sync(device); optimization_seconds = time.perf_counter()-start
        # 训练曲线也用eval模式和当轮固定权重，而非训练时变化的权重。
        train_metrics = evaluate(model, loaders['train'], device)
        val_metrics = evaluate(model, loaders['val'], device)
        record = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                  'train': train_metrics, 'val': val_metrics,
                  'optimization_seconds': optimization_seconds,
                  'train_eval_seconds': train_metrics['seconds'], 'val_eval_seconds': val_metrics['seconds'],
                  'train_presentations': samples_seen,
                  'cumulative_train_presentations': samples_seen + (history[-1]['cumulative_train_presentations'] if history else 0)}
        history.append(record)
        if val_metrics['macro_f1'] > best_f1:
            best_f1, best_epoch = val_metrics['macro_f1'], epoch
            torch.save({'state_dict': model.state_dict(), 'config': config,
                        'families': FAMILIES, 'preprocess': PREPROCESS, 'best_epoch': epoch}, args.out/'best.pt')
        scheduler.step()
        write_json(args.out/'history.json', history)
        with (args.out/'history.csv').open('w', newline='', encoding='utf-8') as handle:
            rows = [{'epoch': r['epoch'], 'train_accuracy': r['train']['accuracy'], 'val_accuracy': r['val']['accuracy'],
                     'train_loss': r['train']['loss'], 'val_loss': r['val']['loss'],
                     'optimization_seconds': r['optimization_seconds'],
                     'train_eval_seconds': r['train_eval_seconds'], 'val_eval_seconds': r['val_eval_seconds'],
                     'cumulative_train_presentations': r['cumulative_train_presentations']} for r in history]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(f"Epoch {epoch}/{args.epochs}: train={train_metrics['accuracy']:.2%}, val={val_metrics['accuracy']:.2%}, "
              f"val macro-F1={val_metrics['macro_f1']:.3f}, optimization={optimization_seconds:.2f}s", flush=True)
    checkpoint = torch.load(args.out/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    splits = ['train', 'val'] if args.skip_test else ['train', 'val', 'test']
    results = {s: evaluate(model, loaders[s], device) for s in splits}
    bench = benchmark(model, device)
    history_plot(history, args.out); final_plot(results, args.out)
    report = {**metadata, 'best_epoch': best_epoch, 'final': results, 'inference_benchmark': bench,
              'optimization_seconds_total': sum(r['optimization_seconds'] for r in history),
              'cumulative_train_presentations': history[-1]['cumulative_train_presentations'],
              'checkpoint_mib': (args.out/'best.pt').stat().st_size/1024**2,
              'wall_seconds': time.perf_counter()-start_all,
              'wall_scope': 'data validation + plots + training + eval + checkpoint IO + benchmark; excludes imports/model initialization',
              'test_evaluated': not args.skip_test,
              'limitations': '已知恶意软件9类分类；不能判断良恶性。单次小样本划分；轻量化精度需多种子验证。'}
    write_json(args.out/'metrics.json', report)
    print(json.dumps({'best_epoch': best_epoch, 'final_accuracy': {s: m['accuracy'] for s,m in results.items()},
                      'wall_seconds': report['wall_seconds']}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True, help='包含dataset.csv及images的目录')
    p.add_argument('--out', type=Path, required=True, help='必须为新的实验目录')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--embed-dim', type=int, default=32)
    p.add_argument('--depths', type=int, nargs=3, default=[2, 2, 2])
    p.add_argument('--heads', type=int, nargs=3, default=[2, 4, 8])
    p.add_argument('--mlp-ratio', type=float, default=4.0)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--drop-path', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.05)
    p.add_argument('--skip-test', action='store_true', help='调参/轻量化试验期间不评估测试集')
    run(p.parse_args())


if __name__ == '__main__':
    main()
