import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from scipy.misc import imresize

SIZE = 128


PATH = '/home/stopa323/Projects/GMM/data/jaffe'
onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

for f in onlyfiles:
    im = Image.open('../data/jaffe/%s' % f)
    img = np.array(im)

    img = imresize(img, (SIZE, SIZE))
    im = Image.fromarray(img)
    im.save('%s/128/%s' % (PATH, f))
