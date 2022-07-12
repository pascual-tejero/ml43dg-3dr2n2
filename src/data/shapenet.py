import typing as t
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
import trimesh
from PIL.Image import Image
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import adjust_hue


class ColoredShapeNet(Dataset):
    """
    Dataset for loading colored ShapeNet dataset
    """

    def __init__(
        self,
        dataset_path: str,
        num_renders: int,
        split: t.Optional[str] = None,
    ):
        """
        :@param num_sample_points: number of points to sample for sdf values per shape
        :@param num_shifts: number of colorized shifts
        :@param num_renders: number of renders per batch
        :@param split: one of 'train', 'val' or 'overfit' - for training,
                      validation or overfitting split
        :@param with_rays: If True each sample will contain ray points
        """
        super().__init__()

        self.num_renders = num_renders
        self.dataset_path = Path(dataset_path)
        self.items = Path(split).read_text().splitlines()

    def __getitem__(self, index):
        """
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of sdf data corresponding to the shape. In particular, this dictionary has keys
                 "name", shape_identifier of the shape
                 "indices": index parameter
                 "points": a num_sample_points x 3  pytorch float32 tensor containing sampled point coordinates
                 "sdf", a num_sample_points x 1 pytorch float32 tensor containing sdf values for the sampled points
        """

        # get shape_id at index
        shape_name = self.items[index]

        cameras, images, silhouettes, depth_maps, _, _, _ = self.get_renders(shape_name)

        sample = {
            "name": shape_name,  # identifier of the shape
            "index": index,  # index parameter
            # ↓ [num_renders, img_width, img_height, 3]
            "images": images,
            # ↓ [num_renders]
            "cameras": cameras,
            # ↓ [num_renders, img_width, img_height, 1]
            "silhouettes": silhouettes,
            # ↓ [num_renders, img_width, img_height, 1]
            "depth_maps": depth_maps,
        }

        return sample

    def __len__(self):
        """
        :return: length of the dataset
        """

        return len(self.items)

    @staticmethod
    def hue_shift(
        colors: torch.Tensor, index: int, num_color_shifts: int
    ) -> torch.Tensor:
        color_idx = index % num_color_shifts

        # transform color
        hue_factor = color_idx / num_color_shifts - 0.5

        colors = (
            adjust_hue(
                colors[None].permute(2, 0, 1),
                hue_factor,
            )
            .squeeze()
            .permute(1, 0)
        )

        return colors

    def get_renders(
        self,
        shape_id: str,
        renders_idx: t.Optional[t.List[int]] = None,
    ) -> t.Tuple[
        FoVPerspectiveCameras,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        t.List[float],
        t.List[float],
        t.List[float],
    ]:
        """
        Return renders for specific shape_id

        @param shape_id: Id of shape for which to extract renders. Random sample if None
        @param renders_idx: Indices of renders
        @return: Tuple of elements:
                    - Cameras: FoVPerspectiveCameras for each render
                    - Images: Tensor of size (N, H, W, 3)
                    - Silhouettes: Tensor of size (N, H, W)
                    - Depth maps: Tensor of size (N, H, W)
                    - Elev: Elevation angle values for renders
                    - Azim: Azimuth angle values for renders
                    - Dists: Distances from where render was made
        """
        path_to_scans = self.dataset_path / str(shape_id) / "scans"
        scans_dirs = [f.name for f in path_to_scans.iterdir() if f.is_dir()]

        if renders_idx is None:
            scans_dirs = np.random.choice(
                scans_dirs, size=self.num_renders, replace=False
            )
        else:
            scans_dirs = [scans_dirs[i] for i in renders_idx]

        elev = []
        azim = []
        depth_maps = []
        dists = []
        images = []
        silhouettes = []

        for scan_dir in scans_dirs:
            path_to_scan = path_to_scans / scan_dir
            scan_meta = torch.load(path_to_scan / "meta.pt")
            scan_image = TF.to_tensor(Image.open(path_to_scan / "image.jpg")).permute(
                1, 2, 0
            )

            # unwrap scan data
            elev.append(scan_meta["elev"])
            azim.append(scan_meta["azim"])
            dists.append(scan_meta["dist"])
            silhouettes.append(scan_meta["silhouette"][None])
            depth_maps.append(scan_meta["depth_map"][None])
            images.append(scan_image[None])

        R, T = look_at_view_transform(dist=dists, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(R=R, T=T)

        images = torch.cat(images)
        silhouettes = torch.cat(silhouettes)
        depth_maps = torch.cat(depth_maps)

        return cameras, images, silhouettes, depth_maps, elev, azim, dists

    def get_mesh(self, shape_id):
        """
        Loading a mesh from the shape with identifier

        :param shape_id: shape identifier for ShapeNet object
        :return: trimesh object representing the mesh
        """
        return trimesh.load(
            self.dataset_path / shape_id / "models" / "mesh.obj", force="mesh"
        )

    def get_scene(self, shape_id):
        """
        Loading a trimesh Scene instance from the shape with identifier

        :param shape_id: shape identifier for ShapeNet object
        :return: trimesh.Scene object representing the mesh
        """
        return trimesh.load(
            self.dataset_path / shape_id / "models" / "model_normalized.obj"
        )


class EmptyDataset(Dataset):
    """
    Mock dataset for DeppSDF empty validation step
    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return {
            "name": torch.tensor([1]),
            "indices": torch.tensor([1]),
            "points": torch.tensor([1]),
            "sdf": torch.tensor([1]),
            "color": torch.tensor([1]),
        }

    def __len__(self):
        return 10


class ShapeNetDataModule(pl.LightningDataModule):
    train_dataset: ColoredShapeNet
    val_dataset: EmptyDataset

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        *,
        num_renders: int = 100,
        path_to_split: str,
        path_to_dataset: str,
    ):
        super().__init__()

        self.num_renders = num_renders

        self.path_to_dataset = path_to_dataset
        self.path_to_split = path_to_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: t.Optional[str] = None) -> None:
        self.train_dataset = ColoredShapeNet(
            dataset_path=self.path_to_dataset,
            num_renders=self.num_renders,
            split=self.path_to_split,
        )

        self.val_dataset = EmptyDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=1, shuffle=False)
