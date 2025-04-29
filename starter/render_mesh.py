"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    R,T,cow_path="data/cow.obj", image_size=256, color1=[0.7, 0.7, 0],color2=[0.7, 0.7, 0],device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    z = vertices[:,:,2]
    z_min = torch.min(vertices[0,:,2])
    z_max = torch.max(vertices[0,:,2])
    color1 = torch.tensor(color1).view(1,3)
    color2 = torch.tensor(color2).view(1,3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color1)  # (1, N_v, 3)

    alpha = (z - z_min) / (z_max - z_min)
    var_color = torch.matmul(alpha.view(2930,1), color2) + torch.matmul(1 - alpha.view(2930,1), color1)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),

    )
    mesh = mesh.to(device)

    # Prepare the camera:

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R, T, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    plt.imsave(args.output_path, image)
