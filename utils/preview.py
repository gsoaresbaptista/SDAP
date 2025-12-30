import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


def plot_source_target(
    source_centroids,
    source_classes,
    target_features,
    target_labels,
    title="",
    save_path=None,
):
    tsne = TSNE(n_components=2, random_state=42)
    all_data = torch.cat([source_centroids, target_features], dim=0).cpu().numpy()
    reduced_data = tsne.fit_transform(all_data)

    source_reduced = reduced_data[: len(source_centroids)]
    target_reduced = reduced_data[len(source_centroids) :]
    plt.figure(figsize=(10, 8))

    for feature, label in zip(target_reduced, target_labels):
        plt.scatter(
            feature[0],
            feature[1],
            color=f"C{label % 10}",
            marker="o",
            s=50,
            alpha=0.6,
            label=None,
        )

    for idx, centroid in enumerate(source_reduced):
        plt.scatter(
            centroid[0],
            centroid[1],
            color=f"C{source_classes[idx] % 10}",
            edgecolor="black",
            linewidth=1.5,
            marker="^",
            s=300,
            label=None,
            alpha=1.0,
            zorder=3,
        )

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
