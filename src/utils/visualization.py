import matplotlib.pyplot as plt


def plot_weight_distribution(model, layer_indices=None):
    if layer_indices is None:
        layer_indices = list(range(len(model.layers)))

    n_plots = len(layer_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, layer_indices):
        ax.hist(model.layers[idx].W.flatten(), bins=30, edgecolor='black')
        ax.set_title(f'Layer {idx} — Weight Distribution')
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def plot_gradient_distribution(model, layer_indices=None):
    if layer_indices is None:
        layer_indices = list(range(len(model.layers)))

    n_plots = len(layer_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, layer_indices):
        if model.layers[idx].dW is None:
            ax.set_title(f'Layer {idx} — No gradients yet')
            ax.axis('off')
            continue
        ax.hist(model.layers[idx].dW.flatten(), bins=30, edgecolor='black')
        ax.set_title(f'Layer {idx} — Gradient Distribution')
        ax.set_xlabel('Gradient')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()
