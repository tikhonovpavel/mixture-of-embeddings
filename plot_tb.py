"""Plot TensorBoard logs for split/merge runs (local only)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: tensorboard. Install it (e.g. `pip install tensorboard`) "
        "or run this script in the same environment where TensorBoard works."
    ) from e


def load_local(tb_path, tags):
    """Load scalar data from a local TensorBoard log directory."""
    ea = EventAccumulator(tb_path)
    ea.Reload()
    result = {}
    for tag in tags:
        events = ea.Scalars(tag)
        result[tag] = {
            'steps': [e.step for e in events],
            'vals': [e.value for e in events],
        }
    return result


def get(data, tag):
    steps = np.array(data[tag]['steps'])
    vals = np.array(data[tag]['vals'])
    return steps, vals


def ema(v, alpha=0.1):
    """Exponential moving average."""
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out


def find_lr_changes(data, lr_tag='lr/backbone'):
    """Find steps where learning rate changes."""
    steps = np.array(data[lr_tag]['steps'])
    vals = np.array(data[lr_tag]['vals'])

    changes = []
    for i in range(1, len(vals)):
        # Detect actual changes (not just numerical noise)
        if abs(vals[i] - vals[i-1]) / (vals[i-1] + 1e-10) > 0.01:  # >1% change
            changes.append(steps[i])
    return changes


def plot_with_ema(ax, steps, vals, color, label, linewidth=1.2, linestyle='-', alpha_raw=0.2, alpha_ema=0.15):
    """Plot raw data faintly + EMA on top."""
    ax.plot(steps, vals, color=color, alpha=alpha_raw, linewidth=0.5, linestyle=linestyle)
    ax.plot(steps, ema(vals, alpha_ema), color=color, label=label, linewidth=linewidth, linestyle=linestyle)


def main():
    run = 'results_split_merge/sm_K8_M8_Qwen_Qwen2.5-3B_20260406_154500'
    base = os.path.abspath(run)

    tags = [
        'train/loss', 'train/loss_a', 'train/loss_b',
        'lambada/acc_avg', 'lambada/acc_a', 'lambada/acc_b',
        'lambada/ce_a', 'lambada/ce_b',
        'lr/backbone', 'lr/projections',
    ]

    # Only load main — it contains all data (backup is identical up to 1M)
    print('Loading main tb (local)...')
    data = load_local(os.path.join(base, 'tb'), tags)

    # Find resume step: backup ends at 1M
    RESUME_STEP = 1_000_000

    # Find LR change steps (only in second run, after resume)
    all_lr_changes = find_lr_changes(data)
    lr_changes = [step for step in all_lr_changes if step > RESUME_STEP]

    def add_resume_line(ax):
        ax.axvline(RESUME_STEP, color='red', linestyle=':', linewidth=1, alpha=0.7, label='resume')

    def add_lr_change_lines(ax):
        """Add vertical lines for each LR change."""
        for step in lr_changes:
            ax.axvline(step, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Train loss (total)
    ax = axes[0, 0]
    s, v = get(data, 'train/loss')
    plot_with_ema(ax, s, v, 'C0', 'loss')
    add_resume_line(ax)
    add_lr_change_lines(ax)
    ax.set_title('Train Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Train loss_a vs loss_b
    ax = axes[0, 1]
    for suffix, color, label in [('a', 'C0', 'loss_a'), ('b', 'C1', 'loss_b')]:
        s, v = get(data, f'train/loss_{suffix}')
        plot_with_ema(ax, s, v, color, label)
    add_resume_line(ax)
    add_lr_change_lines(ax)
    ax.set_title('Train Loss (a vs b)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # LAMBADA accuracy (a and b)
    ax = axes[1, 0]
    for suffix, color, label in [('a', 'C0', 'acc_a'), ('b', 'C1', 'acc_b')]:
        s, v = get(data, f'lambada/acc_{suffix}')
        plot_with_ema(ax, s, v * 100, color, label, alpha_ema=0.25)
    s, v = get(data, 'lambada/acc_avg')
    plot_with_ema(ax, s, v * 100, 'C2', 'acc_avg', alpha_ema=0.25)
    add_resume_line(ax)
    add_lr_change_lines(ax)
    ax.set_title('LAMBADA Accuracy')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # LAMBADA CE (a and b)
    ax = axes[1, 1]
    for suffix, color, label in [('a', 'C0', 'ce_a'), ('b', 'C1', 'ce_b')]:
        s, v = get(data, f'lambada/ce_{suffix}')
        plot_with_ema(ax, s, v, color, label, alpha_ema=0.25)
    add_resume_line(ax)
    add_lr_change_lines(ax)
    ax.set_title('LAMBADA CE Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('CE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Learning rates
    ax = axes[0, 2]
    s, v = get(data, 'lr/backbone')
    ax.plot(s, v, color='C0', label='backbone', linewidth=1.2)
    s, v = get(data, 'lr/projections')
    ax.plot(s, v, color='C1', label='projections', linewidth=1.2)
    add_resume_line(ax)
    add_lr_change_lines(ax)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('LR')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Accuracy ratio (acc_b / acc_a)
    ax = axes[1, 2]
    s_a, v_a = get(data, 'lambada/acc_a')
    s_b, v_b = get(data, 'lambada/acc_b')
    ratio_acc = ema(v_b, 0.25) / np.clip(ema(v_a, 0.25), 1e-8, None)
    ax.plot(s_a, ratio_acc, color='C1', linewidth=1.2, label='acc_b / acc_a')
    # Also show loss ratio for comparison
    # s_la, v_la = get(data, 'train/loss_a')
    # s_lb, v_lb = get(data, 'train/loss_b')
    # ratio_loss = ema(v_la, 0.15) / ema(v_lb, 0.15)
    # ax.plot(s_la, ratio_loss, color='C0', linewidth=1.2, label='loss_a / loss_b')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    add_resume_line(ax)
    add_lr_change_lines(ax)
    ax.set_title('Ratios: Acc vs Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'sm_K8_M8_Qwen_Qwen2.5-3B_20260406_154500\n'
        'red dotted = resume point (LR reset)',
        fontsize=12,
    )
    plt.tight_layout()
    out = 'sm_K8_M8_Qwen_plot.png'
    plt.savefig(out, dpi=150)
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
