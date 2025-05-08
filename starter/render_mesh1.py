"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy
import imageio

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    cow_path, image_size,
    color1, color2,
    camera_R, camera_T
):
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the device.
    device = get_device()

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0).to(device)  # Move to device (N_v, 3) -> (1, N_v, 3)
    z = vertices[:, :, 2]  # Extract z-coordinates
    faces = faces.unsqueeze(0).to(device)  # Move to device (N_f, 3) -> (1, N_f, 3)

    # Compute color gradient based on z-coordinates
    z_min = torch.min(z)
    z_max = torch.max(z)
    alpha = (z - z_min) / (z_max - z_min)  # Normalize z to [0, 1]
    color1 = torch.tensor(color1, device=device).view(1, 1, 3)  # Shape: (1, 1, 3)
    color2 = torch.tensor(color2, device=device).view(1, 1, 3)  # Shape: (1, 1, 3)
    var_color = alpha.unsqueeze(-1) * color2 + (1 - alpha.unsqueeze(-1)) * color1  # Shape: (1, N_v, 3)

    # Apply the color gradient as textures
    textures = pytorch3d.renderer.TexturesVertex(verts_features=var_color)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures).to(device)

    # Prepare the camera
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=camera_R.to(device), T=camera_T.to(device), fov=60, device=device
    )

    # Place a point light in front of the cow
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # Render the image
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend.cpu().numpy()[0, ..., :3]  # (H, W, 3)


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

        image = render_cow(cow_path=args.cow_path, image_size=args.image_size,
                           color1=color1, color2=color2,
                           camera_R=R, camera_T=T)

        my_images.append(numpy.uint8(image[:, :, :] * 255))
        print(i, "/", n_render)
    imageio.mimwrite("images/my_gif.gif", my_images, fps=15)
    plt.imsave(args.output_path, image)
