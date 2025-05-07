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
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    z = vertices[:,:,2]
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color1)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    z_min = torch.min(vertices[0,:,2])
    z_max = torch.max(vertices[0,:,2])
    color1 = torch.tensor(color1).view(1,3)
    color2 = torch.tensor(color2).view(1,3)

    alpha = (z - z_min) / (z_max - z_min)
    var_color = torch.matmul(alpha.view(2930,1), color2) + torch.matmul(1 - alpha.view(2930,1), color1)
        # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
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
    n_render = 60
    color1=[1.0, 0.3, 0.3]
    color2=[0.3, 0.3, 1.0]

    my_images = []
    for i in range(0,n_render):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3, azim=180+360*i/n_render)
        
        image = render_cow(cow_path=args.cow_path, image_size=args.image_size,
                        color1=color1, color2=color2,
                        camera_R=R, camera_T=T)
        
        my_images.append(numpy.uint8(image[:, :, :]*255))
        print(i,"/",n_render)
    imageio.mimwrite("images/my_gif.gif", my_images, fps=15)
    plt.imsave(args.output_path, image)
