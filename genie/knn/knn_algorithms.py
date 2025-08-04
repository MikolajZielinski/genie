from __future__ import annotations

import torch
# import faiss

from dataclasses import dataclass, field
from nerfstudio.configs.base_config import InstantiateConfig
from typing import Type

from genie.knn import optix_knn


@dataclass
class BaseKNNConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: BaseKNN)
    """Base class for KNN configuration."""
    n_neighbours: int = 16
    """Number of nearest neighbours to consider."""
    device: str = 'cuda'
    """Device to run the KNN algorithm on."""


class BaseKNN:
    """Base class for KNN algorithms."""

    def __init__(self, config: BaseKNNConfig):
        super().__init__()
        self.config: BaseKNNConfig = config

    def fit(self, points: torch.Tensor):
        """Fit the KNN model to the given points."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_nearest_neighbours(self, query: torch.Tensor, points: torch.Tensor):
        """Get indices of nearest gaussians."""
        raise NotImplementedError("This method should be implemented by subclasses.")


@dataclass
class TorchKNNConfig(BaseKNNConfig):

    _target: Type = field(default_factory=lambda: TorchKNN)
    """Configuration for Torch KNN algorithm."""
    batch_size: int = 1024
    """Batch size for processing."""

class TorchKNN(BaseKNN):
    """KNN algorithm using PyTorch."""

    def __init__(self, config: TorchKNNConfig):
        super().__init__(config)

    def get_nearest_neighbours(self, query: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Get indices of nearest neighbours using PyTorch."""

        n_coords = query.shape[0]
        nearest_indices = torch.empty((n_coords, self.config.n_neighbours), device=self.config.device, dtype=int)
        distances = torch.empty((n_coords, self.config.n_neighbours), device=self.config.device, dtype=torch.float32)
        for i in range(0, n_coords, self.config.batch_size):
            batch_coords = query[i:i+self.config.batch_size]
            dists = torch.cdist(batch_coords, points).to(device=self.config.device)
            _, batch_nearest_indices = torch.topk(dists, self.config.n_neighbours, largest=False, sorted=False)
            nearest_indices[i:i+self.config.batch_size] = batch_nearest_indices
            distances[i:i+self.config.batch_size] = dists[batch_nearest_indices]
            
        return nearest_indices, distances


# @dataclass
# class FaissKNNConfig(BaseKNNConfig):

#     _target: Type = field(default_factory=lambda: FaissKNN)
#     """Configuration for FAISS KNN algorithm."""

# class FaissKNN(BaseKNN):
#     """KNN algorithm using FAISS."""

#     def __init__(self, config: FaissKNNConfig):
#         super().__init__(config)

#     def get_nearest_neighbours(self, query: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
#         """
#         FAISS KNN using full GPU path and torch.cuda.FloatTensor inputs.

#         Parameters:
#         - coords: (N, D) torch.cuda.FloatTensor

#         Returns:
#         - indices: (N, n_neighbors) torch.LongTensor
#         """
#         assert query.shape[1] == points.shape[1], "Dimension mismatch"
#         assert query.is_cuda and points.is_cuda, "Inputs must be on CUDA"

#         N, D = query.shape

#         # Prepare FAISS
#         res = faiss.StandardGpuResources()

#         # Create CPU index and move to GPU
#         gpu_index = faiss.GpuIndexFlatL2(res, D)

#         # Add means directly
#         gpu_index.add(torch.tensor(points, device=self.config.device))

#         # Search
#         distances, nearest_indices = gpu_index.search(query, self.config.n_neighbours)

#         return nearest_indices, distances
    

# @dataclass
# class FaissIVFKNNConfig(BaseKNNConfig):

#     _target: Type = field(default_factory=lambda: FaissIVFKNN)
#     """Configuration for FAISS IVF KNN algorithm."""
#     nlist: int = 100
#     """Number of Voronoi cells/clusters for IVF (adjust for speed/accuracy tradeoff)."""
    
# class FaissIVFKNN(BaseKNN):
#     """KNN algorithm using FAISS IVF (Inverted File Index)."""

#     def __init__(self, config: FaissIVFKNNConfig):
#         super().__init__(config)

#     def fit(self, points: torch.Tensor):
#         """
#         Fit the IVF index to the given points.

#         Parameters:
#         - points: (M, D) torch tensor (on CPU or CUDA)
#         """
#         # If points has 4 dimensions, drop the last one
#         if points.shape[1] == 4:
#             points = points[:, :3].contiguous()

#         D = points.shape[1]
#         M = points.shape[0]

#         # Prepare FAISS resources
#         self.res = faiss.StandardGpuResources()

#         # Create IVF index
#         quantizer = faiss.GpuIndexFlatL2(self.res, D)
#         self.index_ivf = faiss.GpuIndexIVFFlat(self.res, quantizer, D, self.config.nlist, faiss.METRIC_L2)

#         # Train IVF index on a sample of points
#         train_sample = points[:min(10000, M)]
#         self.index_ivf.train(train_sample)
#         self.index_ivf.add(torch.tensor(points, device=points.device))

#         # Set nprobe (number of cells to search over, higher is more accurate/slower)
#         self.index_ivf.nprobe = min(10, self.config.nlist)

#     def get_nearest_neighbours(self, query: torch.Tensor) -> torch.Tensor:
#         """
#         Search the IVF index for nearest neighbours.

#         Parameters:
#         - query: (N, D) torch tensor (on CPU or CUDA)

#         Returns:
#         - nearest_indices: (N, n_neighbors) torch tensor
#         """
#         assert self.index_ivf is not None, "Index not fitted. Call fit(points) first."
#         assert query.shape[1] == self.index_ivf.d, "Dimension mismatch"

#         distances, nearest_indices = self.index_ivf.search(query, self.config.n_neighbours)
#         return nearest_indices, distances
    

@dataclass
class OptixKNNConfig(BaseKNNConfig):

    _target: Type = field(default_factory=lambda: OptixKNN)
    """Configuration for OptiX KNN algorithm."""
    chi_squared_radius: float = 2.0
    """Chi-squared radius for KNN search."""

class OptixKNN(BaseKNN):
    """KNN algorithm using OptiX."""

    def __init__(self, config: OptixKNNConfig):
        super().__init__(config)

        self.cknn = optix_knn.S_CUDA_KNN()
        optix_knn.CUDA_KNN_Init(config.chi_squared_radius, self.cknn)

    def fit(self, points: torch.Tensor):
        """
        Fit the KNN model to the given points.

        Parameters:
        - points: (M, D) torch tensor (on CUDA)
        """
        assert points.is_cuda, "Points must be on CUDA"
        assert points.shape[1] in [3, 4], "Points must have 3 or 4 dimensions"

        # Expand points to (M, 4) by appending a column of 0.01
        if points.shape[1] == 3:
            pad = torch.full((points.shape[0], 1), 0.01, device=points.device, dtype=points.dtype)
            self.pad_points = torch.cat([points, pad], dim=1)
        else:
            self.pad_points = points

        optix_knn.CUDA_KNN_Fit(self.pad_points, points.shape[0], self.cknn)

    def get_nearest_neighbours(self, query: torch.Tensor) -> torch.Tensor:
        """
        Efficient KNN using OptiX.

        Parameters:
        - query: (N, D) torch tensor (on CUDA)
        - points: (M, D) torch tensor (on CUDA)

        Returns:
        - nearest_indices: (N, n_neighbors) torch tensor
        """
        if query.shape[1] == 3:
            pad = torch.full((query.shape[0], 1), 0.0, device=query.device, dtype=query.dtype)
            self.pad_query = torch.cat([query, pad], dim=1)
        else:
            self.pad_query = query

        distances = torch.empty((self.config.n_neighbours, self.pad_query.shape[0]), dtype=torch.float32, device='cuda')
        indices = torch.empty((self.config.n_neighbours, self.pad_query.shape[0]), dtype=torch.int32, device='cuda')

        optix_knn.CUDA_KNN_KNeighbors(self.pad_query, self.config.n_neighbours, distances, indices, self.cknn)

        distances = distances.T
        indices = indices.T

        return indices, distances
    
    def __del__(self):
        """Clean up resources."""
        optix_knn.CUDA_KNN_Destroy(self.cknn)
        self.cknn = None
        print("OptiX KNN resources cleaned up.")