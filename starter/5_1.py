import argparse
import torch
import numpy as np
import os
import imageio

from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PointsRasterizationSettings, 
    PointsRenderer, 
    PointsRasterizer, 
    AlphaCompositor, 
    look_at_view_transform
)
from pytorch3d.structures import Pointclouds

from starter.render_generic import load_rgbd_data
from starter.utils import unproject_depth_image


def create_pointcloud(rgb, mask, depth, camera, device):
    points, colors = unproject_depth_image(
        torch.tensor(rgb), torch.tensor(mask), torch.tensor(depth), camera)
    return Pointclouds(points=[points.to(device)], features=[colors.to(device)])


def render_pointcloud(pc, camera, image_size=512):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.003,
        points_per_pixel=10
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    return renderer(pc)


def render_and_save_gif(pointcloud, device, output_path, n_frames=60):
    images = []
    for i in range(n_frames):
        azim = 360.0 * i / n_frames
        R, T = look_at_view_transform(dist=4, elev=10.0, azim=azim)
        camera = FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        image = render_pointcloud(pointcloud, camera)[0, ..., :3]
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        images.append(image_np)
        print(f"Rendered frame {i+1}/{n_frames}")

    imageio.mimsave(output_path, images, duration=1000 // 15)
    print(f"Saved gif to {output_path}")


def main(args):
    os.makedirs("output", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据
    data = load_rgbd_data(args.data_path)

    # 固定一个相机用于 unprojection
    R, T = look_at_view_transform(dist=4, elev=0.0, azim=180)
    camera_fixed = FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    # 三个点云
    pc1 = create_pointcloud(data["rgb1"], data["mask1"], data["depth1"], camera_fixed, device)
    pc2 = create_pointcloud(data["rgb2"], data["mask2"], data["depth2"], camera_fixed, device)

    # 合并点云
    combined_points = torch.cat([pc1.points_padded()[0], pc2.points_padded()[0]], dim=0)
    combined_colors = torch.cat([pc1.features_padded()[0], pc2.features_padded()[0]], dim=0)
    pc_combined = Pointclouds(points=[combined_points], features=[combined_colors])

    # 渲染 gif
    
    render_and_save_gif(pc1, device, "output/pointcloud1.gif")
    render_and_save_gif(pc2, device, "output/pointcloud2.gif")
    render_and_save_gif(pc_combined, device, "output/pointcloud_combined.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/rgbd_data.pkl", help="Path to RGB-D dataset")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the rendered image")
    args = parser.parse_args()
    main(args)
