import matplotlib.pyplot as plt
# import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GMM


def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
    plt.show()


digits = load_digits()
print "Shape: %s,%s" % digits.data.shape

plot_digits(digits.data)

# Compression
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)

# Train
gmm = GMM(110, covariance_type='full', random_state=0)
gmm.fit(data)

print(gmm.converged_)

data_new = gmm.sample(100, random_state=0)
digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)
