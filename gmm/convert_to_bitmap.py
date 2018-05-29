from PIL import Image


with open('../urls.txt', 'r') as f:
    for url in f.readlines():
        name = url.split('/')[-1][:-1]

        im = Image.open("../data_raw/%s" % name)
        im = im.convert("RGB")

        new_name = name.replace('.png', '.bmp')
        im.save("../data/%s" % new_name)
