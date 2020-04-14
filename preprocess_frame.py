import numpy as np
import scipy.ndimage


def grayscale(img):
    img = img.mean(axis=2)
    return img


def normalize(img):
    img = (img - 128) / 128 - 1
    return img


def resize(img, scale):
    img = scipy.ndimage.zoom(img, scale, order=1)
    return img


def crop(img, y, x, height, width):
    img = img[y:y+height, x:x+width]
    return img


mspacman_color = np.array([210, 164, 74]).mean()
def preprocess_obs_pacman(obs):
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.mean(axis=2)  # to greyscale
    img[img == mspacman_color] = 0  # improve contrast
    img = (img - 128) / 128 - 1  # normalize from -1. to 1.
    return img.reshape(88, 80, 1)  # reshape and return


if __name__ == '__main__':
    img = np.random.uniform(size=[120, 80, 3])
    print("Grayscale", grayscale(img).shape)
    print("Normalized", normalize(img).shape)
    print("Resized", resize(img, 0.5).shape)
    print("Cropped", crop(img, 20, 20, 100, 60).shape)
    img = np.random.uniform(size=[210, 160, 3])
    print("Pacman", preprocess_obs_pacman(img).shape)
