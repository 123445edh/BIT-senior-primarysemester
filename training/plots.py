"""离线PNG图表。英文图内标签避免不同设备缺少中文字体。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dataset_plot(frame, out):
    counts = frame.groupby(['Class', 'split']).size().unstack(fill_value=0)
    counts = counts.reindex(index=range(1, 10), columns=['train', 'val', 'test'], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts.plot.bar(ax=axes[0], color=['#2667b5', '#28a78b', '#e99836'])
    axes[0].set(title='Unique samples by family', xlabel='Family ID', ylabel='Samples')
    totals = counts.sum()
    axes[1].bar(totals.index, totals.values, color=['#2667b5', '#28a78b', '#e99836'])
    for i, n in enumerate(totals):
        axes[1].text(i, n, str(n), ha='center', va='bottom')
    axes[1].set(title=f'Dataset size: {len(frame)} unique samples', ylabel='Samples', ylim=(0, max(totals)*1.15))
    save(fig, out / 'dataset_counts.png')


def history_plot(history, out):
    if not history:
        return
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for split, color in [('train', '#2667b5'), ('val', '#28a78b')]:
        axes[0, 0].plot(epochs, [r[split]['accuracy']*100 for r in history], label=split, color=color)
        axes[0, 1].plot(epochs, [r[split]['loss'] for r in history], label=split, color=color)
    axes[0, 0].set(title='Accuracy (eval mode, frozen epoch weights)', ylabel='Accuracy (%)', ylim=(0, 100))
    axes[0, 1].set(title='Unweighted cross-entropy', ylabel='Loss')
    bottom = np.zeros(len(history))
    for name in ['optimization_seconds', 'train_eval_seconds', 'val_eval_seconds']:
        values = np.array([r[name] for r in history])
        axes[1, 0].bar(epochs, values, bottom=bottom, label=name.replace('_seconds', ''))
        bottom += values
    axes[1, 0].set(title='Epoch compute + loading time', ylabel='Seconds')
    axes[1, 1].plot(epochs, [r['cumulative_train_presentations'] for r in history], color='#2667b5', label='optimization sample presentations')
    axes[1, 1].set(title='Cumulative training workload (includes repeats)', ylabel='Sample presentations')
    for ax in axes.flat:
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(alpha=.2)
        if len(epochs) <= 20:
            ax.set_xticks(epochs)
    save(fig, out / 'learning_curves.png')


def final_plot(results, out):
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(names))
    for shift, metric, color in [(-.18, 'accuracy', '#2667b5'), (.18, 'macro_f1', '#28a78b')]:
        vals = [results[n][metric]*100 for n in names]
        axes[0].bar(x+shift, vals, .35, label=metric, color=color)
        for pos, value in zip(x+shift, vals):
            axes[0].text(pos, value+1, f'{value:.1f}', ha='center', fontsize=8)
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 110), title='Best validation checkpoint', ylabel='Percent')
    axes[0].legend()
    axes[1].bar(names, [results[n]['seconds'] for n in names], color='#e99836')
    axes[1].set(title='Final evaluation wall time', ylabel='Seconds (includes data loading)')
    axes[2].bar(names, [results[n]['samples_per_second'] for n in names], color='#7864bc')
    axes[2].set(title='Final evaluation throughput', ylabel='Samples / second')
    save(fig, out / 'final_metrics.png')
    if 'test' in results:
        cm = np.array(results['test']['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap='Blues')
        for (i, j), n in np.ndenumerate(cm):
            ax.text(j, i, str(n), ha='center', va='center', color='white' if n > cm.max()/2 else 'black')
        ax.set(xticks=range(9), yticks=range(9), xticklabels=range(1, 10), yticklabels=range(1, 10),
               xlabel='Predicted family ID', ylabel='True family ID', title='Test confusion matrix')
        save(fig, out / 'test_confusion_matrix.png')
