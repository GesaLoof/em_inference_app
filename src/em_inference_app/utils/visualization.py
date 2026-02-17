from PIL import Image
import matplotlib.pyplot as plt
import os

def save_visualization(image: Image.Image, mask: Image.Image, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Prediction")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(mask, cmap="Reds", alpha=0.4)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"vis_mask_{filename}")
    plt.savefig(out_path)
    plt.close()
    return out_path
