import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
import torch as t
import pandas as pd


def img_tile(imgs, iteration, title_prefix, save=False, aspect_ratio=1.0, border=1, border_color=0):
    """
    Visualize the WGAN result for each step
    :param imgs: Numpy array of the generated images
    :param path: Path to save visualized results for each epoch
    :param epoch: Epoch index
    :param save: Boolean value to determine whether you want to save the result or not
    """

    if imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    tile_shape = None
    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break

            # -1~1 to 0~.1
            img = imgs[img_idx]

            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img

    path_name = "figures/%s_iteration%03d" % (title_prefix, iteration) + ".jpg"

    ##########################################
    # Change code below if you want to save results using PIL
    ##########################################
    # tile_img = cv2.resize(tile_img, (1024, 1024))
    plt.title(title_prefix+f", Iterations: {iteration}")
    plt.imshow((tile_img).astype(np.uint8))
    plt.show()
    if save:
        plt.savefig(path_name)

def make_plots(time_series,real_series=None,full_series=None):
    x = np.arange(1, time_series.shape[1]+1)
    plot_array = []
    for index in range(time_series.shape[0]):
        ts = time_series[index,:]
        if real_series is not None:
            rs = real_series[index,:]
        fig = plt.figure(figsize=(2.56, 2.56),dpi=1000)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.plot(x, ts)
        if real_series is not None:
            ax.plot(x,rs,'r-')
        fig.tight_layout()
        canvas.draw()  # draw the canvas, cache the renderer
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plot_array.append(image)
        plt.close(fig)
    return np.array(plot_array)

# fig = plt.figure()
# canvas = FigureCanvas(fig)
# ax = fig.gca()
# x = np.arange(10)
# y = np.arange(10)
# ax.plot(x, y)
# fig.tight_layout()
# canvas.draw()       # draw the canvas, cache the renderer
#
# width, height = fig.get_size_inches() * fig.get_dpi()
# image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
#
# axis_image = plt.imshow(image)
# axis_image.axes.axis('off')
# plt.show()
