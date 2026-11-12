import matplotlib.pyplot as plt


def denormalize_image(image, mean, std):
    """
    Denormalizes a tensor image by reversing the normalization.

    Args:
        image (Tensor): Normalized image tensor.
        mean (list): List of means for each channel.
        std (list): List of standard deviations for each channel.

    Returns:
        Tensor: Denormalized image tensor.
    """
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image


def show_image_by_id(dataloader, image_id, mean, std):
    """
    Displays an image from the DataLoader corresponding to the specified ID.

    Args:
        dataloader (DataLoader): The DataLoader object to search within.
        image_id (int): The index of the image to display.
        mean (list): List of means for denormalization.
        std (list): List of standard deviations for denormalization.
    """
    current_id = 0

    for images, _ in dataloader:
        batch_size = images.size(0)
        if current_id <= image_id < current_id + batch_size:
            image = images[image_id - current_id]
            image = denormalize_image(image, mean, std)
            image = image.permute(1, 2, 0).cpu().numpy()
            plt.imshow(image)
            plt.axis('off')
            plt.show()
            return
        current_id += batch_size
    print(f'Image ID {image_id} not found in the DataLoader.')
