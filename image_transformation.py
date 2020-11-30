# -*- coding: utf-8 -*-

"""some code fork from:
[ModelsGenesis]: https://github.com/MrGiovanni/ModelsGenesis
[uda]: https://github.com/google-research/uda
"""

import numpy as np
import random
import copy
from scipy.special import comb

from skimage import transform
import torchvision


class ImageTransforms:
    def __init__(self, image_size=224, rescale_size=256, train=True):
        self.train = train

        self.train_transform = torchvision.transforms.Compose(
            [
                Resize(rescale_size=rescale_size, image_size=image_size),
                RandomFlip(),
                RandomLocalPixelShuffling(),
                RandomNonlinearTransformation(),
                RandomPainting(),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                Resize(rescale_size=rescale_size, image_size=image_size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        if self.train:
            return self.train_transform(x), self.train_transform(x)
        else:
            return self.test_transform(x)


class Resize:
    def __init__(self, rescale_size=256, image_size=224):
        self.rescale_size = rescale_size
        self.image_size = image_size

    def __call__(self, img):
        rows, cols = img.shape[0], img.shape[1]

        if rows > cols:
            new_rows, new_cols = self.rescale_size * rows / cols, self.rescale_size
        else:
            new_rows, new_cols = self.rescale_size, self.rescale_size * cols / rows
        new_rows, new_cols = int(new_rows), int(new_cols)

        img = transform.resize(img, (new_rows, new_cols))

        top = (new_rows - self.image_size) // 2
        left = (new_cols - self.image_size) // 2
        img = img[top: self.image_size + top, left: self.image_size + left]

        return img


class RandomFlip:
    def __call__(self, img, prob=0.4, cnt=1):
        # augmentation by flipping, cnt=3
        while random.random() < prob and cnt > 0:
            axis = random.choice([0, 1])
            img = np.flip(img, axis=axis).copy()
            cnt -= 1

        return img


class InPainting:
    def __call__(self, img, cnt=5):
        h, w = img.shape[0], img.shape[1]

        new_img = copy.deepcopy(img)
        while cnt > 0 and random.random() < 0.95:
            block_x = random.randint(h // 6, h // 3)
            block_y = random.randint(w // 6, w // 3)
            noise_x = random.randint(3, h - block_x - 3)
            noise_y = random.randint(3, w - block_y - 3)

            painting_val = random.random()
            noise = np.ones((block_x, block_y)) * painting_val
            new_img[noise_x: noise_x + block_x, noise_y: noise_y + block_y] = noise

            cnt -= 1

        return new_img


class OutPainting:
    def __call__(self, img, cnt=4):
        new_img = np.zeros_like(img)
        h, w = img.shape[0], img.shape[1]

        while cnt > 0 and random.random() < 0.95:
            block_x = h - random.randint(3 * h // 7, 4 * h // 7)
            block_y = w - random.randint(3 * w // 7, 4 * w // 7)
            noise_x = random.randint(3, h - block_x - 3)
            noise_y = random.randint(3, w - block_y - 3)

            noise = img[noise_x: noise_x + block_x, noise_y: noise_y + block_y]
            new_img[noise_x: noise_x + block_x, noise_y: noise_y + block_y] = noise

            cnt -= 1

        return new_img


class RandomPainting:
    def __call__(self, img, prob=0.5):
        """prob=0.9"""
        if random.random() >= prob:
            return img

        if random.random() < 0.2:
            painting = InPainting()
        else:
            painting = OutPainting()

        new_img = painting(img)

        return new_img


class RandomLocalPixelShuffling:
    def __call__(self, img, prob=0.25):
        """prob=0.5"""
        if random.random() >= prob:
            return img

        rows, cols = img.shape[0], img.shape[1]
        new_img = copy.deepcopy(img)

        n_block = 1000
        for _ in range(n_block):
            block_x = random.randint(1, rows // 10)
            block_y = random.randint(1, cols // 10)

            noise_x = random.randint(0, rows - block_x)
            noise_y = random.randint(0, cols - block_y)

            window = img[noise_x: noise_x + block_x, noise_y: noise_y + block_y]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_x, block_y))

            new_img[noise_x: noise_x + block_x, noise_y: noise_y + block_y] = window

        return new_img


class RandomNonlinearTransformation:
    """The pixel range should be limited within [0, 1]"""
    def __call__(self, img, prob=0.5):
        """prob=0.9"""
        if random.random() >= prob:
            return img

        points = [[0, 0],
                  [random.random(), random.random()],
                  [random.random(), random.random()],
                  [1, 1]]

        x_vals, y_vals = self.bezier_curve(points, n_times=1000)
        if random.random() < 0.5:
            # Half change to get flip
            x_vals = np.sort(x_vals)
        else:
            x_vals, y_vals = np.sort(x_vals), np.sort(y_vals)
        nonlinear_img = np.interp(img, x_vals, y_vals)

        return nonlinear_img

    def bernstein_poly(self, i, n, t):
        """The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, n_times=100):
        """Given a set of control points, return the bezier curve defined by the
        control points. Control points should be a list of lists, or list of tuples
        such as [[1,1], [2,3], [4,5], ..., [Xn, Yn]]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
        """
        n_points = len(points)
        x_points = np.array([p[0] for p in points])
        y_points = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, n_times)

        polynomial_array = np.array(
            [self.bernstein_poly(i, n_points - 1, t) for i in range(0, n_points)]
        )

        x_vals = np.dot(x_points, polynomial_array)
        y_vals = np.dot(y_points, polynomial_array)

        return x_vals, y_vals


class RandomShift:
    """Forked from: https://github.com/google-research/uda/blob/
    960684e363251772a5938451d4d2bc0f1da9e24b/image/randaugment/
    augmentation_transforms.py#L64
    """
    def __call__(self, img, shift=20, prob=0.5):
        """Zero pad by `amount` zero pixels on each side then take a random crop.
        Args:
            img: numpy image that will be zero padded and cropped.
            shift: amount of zeros to pad `img` with horizontally and verically.
        Returns:
            The cropped zero padded img. The returned numpy array will be of the
            same shape as `img`.
        """
        rows, cols = img.shape[0], img.shape[1]

        padded_img = np.zeros((rows + shift * 2, cols * shift * 2))
        padded_img[shift: rows + shift, shift: cols + shift] = img

        top = random.randint(0, 2 * shift)
        left = random.randint(0, 2 * shift)

        new_img = padded_img[top: top + rows, left: left + cols]

        return new_img


class RandomCutout:
    """Fork from: https://github.com/google-research/uda/blob/
    960684e363251772a5938451d4d2bc0f1da9e24b/image/randaugment/
    augmentation_transforms.py#L85

    Apply cutout with mask of shape `size` x `size` to `img`.

    The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
    This operation applies a `size`x`size` mask of zeros to a random location
    within `img`.

    Args:
        img: Numpy image that cutout will be applied to.
        size: Height/width of the cutout mask that will be

    Returns:
        A numpy tensor that is the result of applying the cutout mask to `img`.
    """
    def __call__(self, img, cutout_size=8):
        rows, cols = img.shape[0], img.shape[1]

        mask, _, _ = self.create_cutout_mask(rows, cols, cutout_size)
        new_img = img * mask

        return new_img

    def create_cutout_mask(self, img_height, img_width, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.

        Args:
            img_height: Height of image cutout mask will be applied to.
            img_width: Width of image cutout mask will be applied to.
            size: Size of the zeros mask.

        Returns:
            A mask of shape `img_height` x `img_width` with all ones except for a
            square of zeros of shape `size` x `size`. This mask is meant to be
            elementwise multiplied with the original image. Additionally returns
            the `upper_coord` and `lower_coord` which specify where the cutout mask
            will be applied.
        """
        # Sample center where cutout mask will be applied
        height = random.randint(0, img_height)
        width = random.randint(0, img_width)

        # Determine upper right and lower left corners of patch
        upper_coord = (max(0, height - size // 2),
                       max(0, width - size // 2))
        lower_coord = (min(img_height, height + size // 2),
                       min(img_width, width + size // 2))
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width))
        zeros = np.zeros((mask_height, mask_width))
        mask[upper_coord[0]: lower_coord[0], upper_coord[1]: lower_coord[1]] = zeros

        return mask, upper_coord, lower_coord


def main():
    print("Transformation ...")
    img = np.load("/home/zhai/Projects/data/views/1.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208_0_1.npy")
    print(f"img size: {img.shape}")
    # trans = torchvision.transforms.Resize()
    # trans = Resize()
    # trans = RandomFlip()
    # trans = InPainting()
    # trans = OutPainting()
    # trans = RandomPainting()
    # trans = RandomLocalPixelShuffling()
    # img = trans(img)
    # trans = RandomNonlinearTransformation()
    # trans = ImageTransforms(image_size=224, rescale_size=224, train=True)
    # trans = RandomShift()
    trans = RandomCutout()

    new_img = trans(img)

    print(f"new img size: {new_img.shape}")
    print(f"img: {new_img}")
    print(f"pixel: {new_img.max(), new_img.min()}")

    import matplotlib.pyplot as plt

    # plt.imshow(img, cmap="gray")
    # plt.show()
    # img = transform.resize(img, output_shape=(224, 224))
    # new_img = np.flip(img, axis=0)
    # print(new_img.shape)
    plt.imshow(new_img, cmap="gray")
    plt.show()

    # plt.imshow(mask[:, :, 0], cmap="gray")
    # plt.show()


if __name__ == '__main__':
    main()
