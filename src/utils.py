import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

# Crop the bounding box around the digit and resize to fixed size.
def crop_bounding_box(image, resize_shape=(28, 28), threshold=30, margin=0):
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Binarize to find digit pixels
    mask = image > threshold
    coords = np.argwhere(mask)

    # no image = black
    if coords.size == 0:
        return np.full(resize_shape, -1.0) 
    else:
        # Find bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)  

        # Add margin (clamp to image bounds)
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(image.shape[0]-1, y_max + margin)
        x_max = min(image.shape[1]-1, x_max + margin)

        cropped = image[y_min:y_max+1, x_min:x_max+1] # Crop digit

    # Resize to fixed shape
    if cropped.ndim == 2:
        cropped = cropped[..., np.newaxis]
    cropped_resized = tf.image.resize(cropped, resize_shape, method="nearest").numpy().squeeze()
    cropped_resized = np.where(cropped_resized > 0.5, 1, -1) # Binarize after cropping
    return cropped_resized


# Find the mean of image pixels of each digit
def build_mean_imagepx(images, labels, resize_shape=(28,28), threshold=30, margin_draw=2):

    memory_pattern = []  # 10 patterns for each digit 
    draw_patterns = [] 
    for digit in range(10):
        digit_imgs = images[labels == digit][:1]

        # Visualization pattern (with margin)
        draw_img = crop_bounding_box(digit_imgs[0], resize_shape=resize_shape, threshold=threshold, margin=margin_draw)
        draw_patterns.append(draw_img)

        # Training pattern (no margin)
        cropped_resized_imgs = [crop_bounding_box(img, resize_shape=resize_shape, threshold=threshold, margin=0) for img in digit_imgs]
        cropped_resized_imgs = [np.where(img > 0, 1, -1).flatten() for img in cropped_resized_imgs if img.shape == resize_shape]

        mean_imgpx = np.mean(cropped_resized_imgs, axis=0) # Average the bipolar cropped images
        binarized_proto = np.where(mean_imgpx > 0, 1, -1)  # Use 0 for bipolar threshold 
        memory_pattern.append(binarized_proto)

    return np.array(memory_pattern), np.array(draw_patterns)


# plot digit
def plot_digits(flattened_images, shape=(28, 28), titles=None):
    # Plot a single image
    if flattened_images.ndim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(flattened_images.reshape(shape), cmap='gray')
        ax.axis("off")
        if titles and len(titles) > 0:
            ax.set_title(titles[0])
        plt.show()
    
    # Plot a grid of images
    else:
        n_images = flattened_images.shape[0]
        cols = 3                             
        rows = (n_images + cols - 1) // cols 

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.ravel()                  # Flatten

        for i in range(n_images):
            axes[i].imshow(flattened_images[i].reshape(shape), cmap='gray')
            axes[i].axis("off")
            if titles and i < len(titles):
                axes[i].set_title(titles[i])

        # Hide any unused subplots
        for j in range(n_images, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()