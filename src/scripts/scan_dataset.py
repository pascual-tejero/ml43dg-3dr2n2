import argparse
import os
import traceback
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils.scan_obj import generate_renders

try:
    set_start_method("spawn", force=True)
except Exception:
    ...

LOG_FILE = "./dataset_scan_log.txt"


def scan_sample(input):
    try:
        (
            model_id,
            dataset_path,
            obj_prefix,
            num_views,
            image_size,
            dist,
            device_name,
            num_batch,
            voxelize,
        ) = input

        device = torch.device(device_name)

        if device_name.startswith("cuda"):
            torch.cuda.empty_cache()

        # creat renders
        obj_path = Path(dataset_path) / model_id
        mesh_path = str(obj_path / obj_prefix)

        with torch.no_grad():
            renders = generate_renders(
                mesh_path=mesh_path,
                num_views=num_views,
                image_size=image_size,
                device=device,
                target_device=torch.device("cpu"),
                num_batch=num_batch,
                dist=dist,
            )
        if device_name.startswith("cuda"):
            torch.cuda.empty_cache()

        if voxelize:
            # create voxel representation
            mesh = o3d.io.read_triangle_mesh(mesh_path)

            # fit to unit cube
            mesh.scale(
                1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
                center=mesh.get_center(),
            )

            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
                mesh,
                voxel_size=0.05,
            )

            coo = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
            voxels = np.zeros((2, 32, 32, 32))
            voxels[0, :] = 1
            voxels[:, coo[:, 0], coo[:, 1], coo[:, 2]] = np.array([0, 1])[:, None]

            # save voxels
            voxels = torch.tensor(voxels)
            torch.save(voxels, str(obj_path / "voxels.pt"))

        # create scan folder
        scans_folder = obj_path / "scans"
        scans_folder.mkdir(exist_ok=True)

        # save scan
        for i in range(renders.images.shape[0]):
            scan_folder = scans_folder / f"render_{i}"
            scan_folder.mkdir(exist_ok=True)

            save_image(
                renders.images[i].permute(2, 0, 1).clone(),
                str(scan_folder / f"image.jpg"),
            )

            torch.save(
                dict(
                    dist=dist,
                    azim=renders.azim[i].clone(),
                    elev=renders.elev[i].clone(),
                    silhouette=renders.silhouette[i].clone(),
                    depth_map=renders.depth_maps[i].clone(),
                ),
                str(scan_folder / f"meta.pt"),
            )

    except Exception:
        with open(LOG_FILE, "a") as log_file:
            traceback.print_exc(file=log_file)


if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument(
        "--num_scans", type=int, help="Number of scans per obj file.", default=100
    )
    parser.add_argument(
        "--voxelize", type=bool, help="Wheatehr or not to create voxels", default=True
    )
    parser.add_argument(
        "--image_size", type=int, help="Size of scan image", default=128
    )
    parser.add_argument(
        "--num_thread", type=int, help="Number of thread used ot colorize", default=2
    )
    parser.add_argument(
        "--device", type=str, help="Torch device to use for render", default="cpu"
    )
    parser.add_argument(
        "--num_batch",
        type=int,
        help="Number of batches foe simultaneous render",
        default=10,
    )
    parser.add_argument(
        "--obj_prefix",
        type=str,
        help="Prefix for obj files",
        default="model_normalized.obj",
    )
    parser.add_argument(
        "--dist",
        type=float,
        help="Distance for renders",
        default=2.7,
    )

    args = parser.parse_args()

    # colorizing
    _, ids, _ = next(os.walk(args.dataset))
    inputs = [
        (
            model_id,
            args.dataset,
            args.obj_prefix,
            args.num_scans,
            args.image_size,
            args.dist,
            args.device,
            args.num_batch,
            args.voxelize,
        )
        for model_id in ids
    ]

    if args.num_thread > 1:
        with Pool(args.num_thread) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(scan_sample, inputs),
                    total=len(inputs),
                    desc="Scanning obj files: ",
                )
            )
    else:
        for i in tqdm(inputs):
            scan_sample(i)
