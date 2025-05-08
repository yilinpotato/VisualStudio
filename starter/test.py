import argparse

import imageio
import numpy
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from  starter.render_mesh import render_cow
from starter.utils import get_device, get_mesh_renderer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    n_render = 60
    color1 = [1.0, 0.3, 0.3]
    color2 = [0.3, 0.3, 1.0]

    my_images = []
    for i in range(0, n_render):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3, azim=180 + 360 * i / n_render)

        image = render_cow(R,T,cow_path=args.cow_path, image_size=args.image_size,
                           color1=color1,color2=color2,
                          )

        my_images.append(numpy.uint8(image[:, :, :] * 255))
        print(i, "/", n_render)

    imageio.mimwrite("output/my_gif.gif", my_images, fps=15)