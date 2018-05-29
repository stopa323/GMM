import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
from sklearn.mixture import GMM


PATH = '/home/stopa323/Projects/GMM/data/jaffe/128'
FILES = [f for f in listdir(PATH) if isfile(join(PATH, f))]


def plot_digits(data):
    fig, ax = plt.subplots(5, 5, figsize=(64, 64),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.01)
    for i, axi in enumerate(ax.flat):
        axi.imshow(data[i].reshape(128, 128), cmap='gray')
    plt.show()


def load_data():
    data = np.zeros((len(FILES), 16384))
    for idx, f in enumerate(FILES):
        im = Image.open('%s/%s' % (PATH, f))
        img = np.array(im)
        img = img.reshape((1, 16384))
        data[idx] = img[0]
    return data


data = load_data()
plot_digits(data)

# Compression
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(data)

# Train
gmm = GMM(10, covariance_type='full', random_state=0)
gmm.fit(data)

print(gmm.converged_)

data_new = gmm.sample(100, random_state=0)
digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)
