import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


def get_images(paths, labels, nb_samples=None):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    image_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]

    return image_labels


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        config={},
        cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_type: train/val/test
            config: data_folder - folder where the data is located
                    img_size - size of the input images
            cache: whether to cache the images loaded
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train : num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val :]
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

        self.sample_fn = random.sample

        self.shuffle_fn = np.random.shuffle

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / image.max()
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image

    def _sample(self, sample_fn=None, shuffle_fn=None):
        """
        Samples a batch for training, validation, or testing
        Args:
            sample_fn - pointer to a sampling function to be used
            shuffle_fn - pointer to a shuffling function to be used
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and is a numpy array
                2. label batch has shape [K+1, N, N] and is a numpy array
            where K is the number of "shots", N is number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.
            
            3. The value for `self.num_samples_per_class` will be set to K+1 
            since for K-shot classification you need to sample K supports and 
            1 query.

            4. PyTorch uses float32 as default for representing model parameters. 
            You would need to return numpy arrays with the same datatype
        """

        if sample_fn is None:
            sample_fn = self.sample_fn

        if shuffle_fn is None:
            shuffle_fn = self.shuffle_fn

        #############################
        ### START CODE HERE ###
        # Sample classes from the dataset
        sampled_classes = sample_fn(self.folders, self.num_classes)
        labels_and_images = get_images(sampled_classes, range(self.num_classes), self.num_samples_per_class)

        # Initialize arrays for images and labels
        images = np.zeros((self.num_samples_per_class, self.num_classes, 784), dtype=np.float32)
        labels = np.zeros((self.num_samples_per_class, self.num_classes, self.num_classes), dtype=np.float32)

        # The arrays with images and one-hot encoded labels
        for class_idx in range(self.num_classes):
            class_images = [img for lbl, img in labels_and_images if lbl == class_idx]
            for i in range(self.num_samples_per_class):
                image_path = class_images[i]
                images[i, class_idx, :] = self.image_file_to_array(image_path, 784)
                labels[i, class_idx, :] = np.eye(self.num_classes, dtype=np.float32)[class_idx]

        # Shuffle the last set of images and labels
        indices = np.arange(self.num_classes)
        shuffle_fn(indices)
        images[-1] = images[-1, indices]
        labels[-1] = labels[-1, indices]

        return images, labels
 
        ### END CODE HERE ###

    def __iter__(self):
        while True:
            yield self._sample()

    def plot_image_label(self, img, label):
        actual_images = img.reshape(*img.shape[:2], *self.img_size)
        fig, axs = plt.subplots(2 * self.num_samples_per_class, self.num_classes, figsize=(16, 10))
        for i_sample in range(self.num_samples_per_class):
            for i_classes in range(self.num_classes):
                imgx = actual_images[i_sample, i_classes]
                axs[2 * i_sample, i_classes].imshow(imgx)
                axs[2 * i_sample, i_classes].axis("off")
                imgx = np.expand_dims(label[i_sample, i_classes], axis=0)
                axs[2 * i_sample + 1, i_classes].imshow(imgx)
                axs[2 * i_sample + 1, i_classes].axis("off")
        fig.show()
