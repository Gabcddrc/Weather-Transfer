from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):

    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring()) 


def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)
