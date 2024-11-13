# This has not been installed yet!
import torch
from ricomodels.utils.losses import dice_loss


def test_dice():
    outputs = torch.tensor(
        [
            [
                # Class 0 scores
                [[0.2, 0.8], [0.6, 0.4]],
                # Class 1 scores
                [[0.8, 0.2], [0.4, 0.6]],
            ]
        ],
        dtype=torch.float32,
    )

    # Define labels tensor with shape [batch_size, H, W]
    labels = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int8)

    expected = 7.0 / 10
    loss = dice_loss(outputs, labels, epsilon=1e-6)
    print(loss)
