import imageio
images = []
filenames = ["lda_figure" + str(i) + '.png' for i in range(0,351, 10)]
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('giffed.gif', images)