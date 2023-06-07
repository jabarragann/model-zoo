from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from monai.visualize.utils import blend_images
from monai.networks.nets import FlexibleUNet

import torchvision.transforms as T


img_transforms = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalize
    ]
)


def inference():
    torch.set_grad_enabled(False)
    workspace = Path(__file__).parent.resolve()

    device = "cuda"
    path_to_weights = (workspace / "../models/myweights.pt").resolve()
    path_to_images = (workspace / "../sample_data/img_000035.png").resolve()

    im = Image.open(path_to_images)
    im = np.array(im)

    # Load model
    model = FlexibleUNet(
        in_channels=3,
        out_channels=5,
        backbone="efficientnet-b0",
        pretrained=True,
        is_pad=False,
    ).to(device)
    model.load_state_dict(torch.load(path_to_weights))
    model.eval()

    # Inference
    input_tensor = img_transforms(im).to(device)
    input_tensor = torch.unsqueeze(input_tensor, 0)  # Add batch dimension. 4D input_tensor
    inferred = model(input_tensor)
    inferred = inferred[0]  # go back to 3D tensor
    inferred_single_ch = inferred.argmax(dim=0, keepdim=True)  # Get a single channel image

    inferred_single_ch = inferred_single_ch.detach().cpu()
    input_tensor = input_tensor.detach().cpu()[0]

    blended = blend_images(input_tensor, inferred_single_ch, cmap="viridis", alpha=0.9).numpy()
    blended = (np.transpose(blended, (1, 2, 0)) * 254).astype(np.uint8)

    # display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.set_tight_layout(True)
    ax1.set_title("Input")
    ax2.set_title("Inferred")
    ax1.axis("off")
    ax2.axis("off")
    ax1.imshow(im)
    ax2.imshow(blended)
    plt.show()


if __name__ == "__main__":
    inference()
