import numpy as np
import cv2


def omni2panoramic(image, x, y, out_r, in_r):
    """ Turn omni image into panorama.

        :param: image: 3D array with omni image data
        :param: x: middle x from the image
        :param: y: middle y from the image

        :return: 3D array of the panorama image
    """
    pi = np.pi
    w = np.floor((out_r + in_r) * pi).astype(int)
    panorama = np.zeros([out_r - in_r, w, image.shape[2]])

    for r, i in zip(range(out_r, in_r, -1), range(out_r - in_r)):
        for v, j in zip(range(w, 0, -1), range(w)):
            coord_x1 = np.floor(r * np.sin(v * 2 * pi / w) + x).astype(int)
            val_x1 = r * np.sin(v * 2 * pi / w) + x - coord_x1
            coord_y1 = np.floor(r * np.cos(v * 2 * pi / w) + y).astype(int)
            val_y1 = r * np.cos(v * 2 * pi / w) + y - coord_y1

            panorama[i][j] = ((2 - val_x1 - val_y1) * image[coord_x1, coord_y1, :] +
                              (1 - val_x1 + val_y1) * image[coord_x1, coord_y1 + 1, :] +
                              (1 + val_x1 - val_y1) * image[coord_x1 + 1, coord_y1, :] +
                              (val_x1 + val_y1) * image[coord_x1 + 1, coord_y1 + 1, :]) / 4

    # return cv2.resize(panorama, dsize=(512, 128), interpolation=cv2.INTER_CUBIC)
    panorama[panorama > 1] = 1
    return panorama
