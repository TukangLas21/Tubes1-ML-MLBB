from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curves(results_dict, title, save_path=None):
    fig = plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    for name, result in results_dict.items():
        plt.plot(result['history']['train_loss'], label=name)
    plt.title(f'{title} - Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for name, result in results_dict.items():
        plt.plot(result['history']['val_loss'], label=name)
    plt.title(f'{title} - Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()


def plot_weight_distribution(model, layer_indices=None, save_path=None):
    if layer_indices is None:
        layer_indices = list(range(len(model.layers)))

    n_plots = len(layer_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, layer_indices):
        ax.hist(model.layers[idx].W.data.flatten(), bins=30, edgecolor='black')
        ax.set_title(f'Layer {idx} — Weight Distribution')
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()


def plot_gradient_distribution(model, layer_indices=None, save_path=None):
    if layer_indices is None:
        layer_indices = list(range(len(model.layers)))

    n_plots = len(layer_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, layer_indices):
        grad = model.layers[idx].W.grad
        if grad is None:
            ax.set_title(f'Layer {idx} — No gradients yet')
            ax.axis('off')
            continue
        ax.hist(grad.flatten(), bins=30, edgecolor='black')
        ax.set_title(f'Layer {idx} — Gradient Distribution')
        ax.set_xlabel('Gradient')
        ax.set_ylabel('Count')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()