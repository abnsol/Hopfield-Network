import numpy as np
import tensorflow_datasets as tfds

def load_data(split="train"):
    ds = tfds.load("mnist", split=split, shuffle_files=True, as_supervised=True)
    images = []
    labels = []
    for img, lbl in tfds.as_numpy(ds):
        images.append(img.squeeze())  
        labels.append(lbl)
    return np.array(images), np.array(labels)

