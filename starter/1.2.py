import argparse
import os
import torch
import numpy as np
import imageio

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    look_at_view_transform,
    TexturesVertex
)


def render_cow_z_colored(cow_path, image_size, color1, color2, camera_R, camera_T):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    mesh = load_objs_as_meshes([cow_path], device=device)

    # 获取顶点 z 坐标
    vertices = mesh.verts_packed().unsqueeze(0)
    z = vertices[:, :, 2]  # shape: [1, V]
    z_min = torch.min(z)
    z_max = torch.max(z)

    # 插值颜色：color = alpha * color2 + (1 - alpha) * color1
    color1 = torch.tensor(color1, device=device).view(1, 3)
    color2 = torch.tensor(color2, device=device).view(1, 3)
    alpha = (z - z_min) / (z_max - z_min + 1e-5)
    interpolated_color = alpha.unsqueeze(-1) * color2 + (1 - alpha.unsqueeze(-1)) * color1

    # 设置颜色贴图
    mesh.textures = TexturesVertex(verts_features=interpolated_color)

    # 摄像头设置
    cameras = FoVPerspectiveCameras(device=device, R=camera_R, T=camera_T)

    # 光照 & 渲染器
    raster_settings = RasterizationSettings(image_size=image_size)
    lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    images = renderer(mesh)
    image = images[0, ..., :3].cpu().numpy()
    return image


def main(args):
    os.makedirs("output", exist_ok=True)
    n_render = 60

    color1 = [0.0, 0.0, 0.5]  # blue (近处)
    color2 = [1.0, 0.0, 0.0]  # red (远处)

    # Dolly Zoom 的 FOV 变化（从15到90度）
    fovs = torch.linspace(15, 60, n_render)
    scene_width = 1.0  # 假设场景宽度恒定为 1

    my_images = []
    for i, fov in enumerate(fovs):
        fov_rad = torch.deg2rad(fov / 2)
        distance = scene_width / (2 * torch.tan(fov_rad))  # 保持目标大小不变所需的相机距离

        R, T = look_at_view_transform(dist=distance.item(), elev=0.0, azim=0.0)

        # 动态设置相机FOV
        image = render_cow_z_colored(
            cow_path=args.cow_path,
            image_size=args.image_size,
            color1=color1,
            color2=color2,
            camera_R=R,
            camera_T=T
        )

        my_images.append(np.uint8(image * 255))
        print(f"Rendered frame {i+1}/{n_render} - FOV: {fov.item():.2f}, Distance: {distance.item():.2f}")

    imageio.mimsave("output/dolly_zoom_colored_cow.gif", my_images, duration=1000 // 15, loop=0)
    print("Saved gif to output/dolly_zoom_colored_cow.gif")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cow_path', type=str, default="data/cow.obj", help="Path to cow.obj")
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()
    main(args)
