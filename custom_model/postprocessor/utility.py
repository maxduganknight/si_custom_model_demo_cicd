"""Library Import."""
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def get_masks(gray_masks):
    """Get binary output of the model ( 512 x 512 x 5) creates individual RGB mask for each segment.

    Args:
        gray_masks (nparray): model's outpout

    Returns:
        final_image (nparray): segmented image arrays
    """
    int1 = gray_masks[:, :, 0]
    prox = gray_masks[:, :, 1]
    mid = gray_masks[:, :, 2]
    dist = gray_masks[:, :, 3]
    int2 = gray_masks[:, :, 4]

    int1 = np.repeat(np.expand_dims(int1, axis=-1), 3, axis=-1)
    int1[:, :, 0] = 0
    int1[:, :, 2] = 0
    prox = np.repeat(np.expand_dims(prox, axis=-1), 3, axis=-1)
    prox[:, :, 1] = 0
    prox[:, :, 2] = 0
    mid = np.repeat(np.expand_dims(mid, axis=-1), 3, axis=-1)
    mid[:, :, 0] = 0
    mid[:, :, 1] = 0
    dist = np.repeat(np.expand_dims(dist, axis=-1), 3, axis=-1)
    dist[:, :, 2] = 0
    int2 = np.repeat(np.expand_dims(int2, axis=-1), 3, axis=-1)
    int2[:, :, 1] = 0

    return (int1, prox, mid, dist, int2)


def plotter(image, seg_masks):
    """Overlay masks on the original image.

    Args:
        image (nparray): original image array
        seg_masks (nparray): mask in nparray

    Returns:
        plt: image with overlaid mask
    """
    fig = plt.figure(figsize=(10, 10))

    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.imshow(seg_masks, cmap="gray", alpha=0.4)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    final_image = np.fromstring(
        fig.canvas.tostring_rgb(), dtype="uint8", sep=""
    ).reshape((height, width, 3))
    final_image = final_image[120:-110, 130:-120]

    return final_image
