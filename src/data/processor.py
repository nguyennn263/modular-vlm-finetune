"""
VinternProcessor: Dynamic High Resolution Image Processing
Triển khai Dynamic Tiling cho Vision-Language Model
"""
import math
from typing import List, Tuple, Optional
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# Các aspect ratio được hỗ trợ (tối đa 12 tiles)
SUPPORTED_ASPECT_RATIOS = [
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 1), (4, 2), (4, 3),
]


class VinternProcessor:
    """Processor cho Dynamic High Resolution tiling"""
    
    def __init__(
        self,
        image_size: int = 448,
        max_tiles: int = 12,
        min_tiles: int = 1,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.max_tiles = max_tiles
        self.min_tiles = min_tiles
        self.mean = mean
        self.std = std
        
        # Transform cho từng tile
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
    
    def find_best_aspect_ratio(
        self, 
        width: int, 
        height: int
    ) -> Tuple[int, int]:
        """Tìm aspect ratio tối ưu cho ảnh dựa trên kích thước gốc"""
        aspect_ratio = width / height
        best_ratio = (1, 1)
        min_diff = float('inf')
        
        for (w_tiles, h_tiles) in SUPPORTED_ASPECT_RATIOS:
            num_tiles = w_tiles * h_tiles
            if num_tiles < self.min_tiles or num_tiles > self.max_tiles:
                continue
            
            ratio = w_tiles / h_tiles
            diff = abs(ratio - aspect_ratio)
            
            # Ưu tiên aspect ratio gần nhất, sau đó là nhiều tiles hơn
            if diff < min_diff or (diff == min_diff and num_tiles > best_ratio[0] * best_ratio[1]):
                min_diff = diff
                best_ratio = (w_tiles, h_tiles)
        
        return best_ratio
    
    def get_tiles(
        self, 
        image: Image.Image, 
        max_num: int = 12
    ) -> Tuple[List[Image.Image], Tuple[int, int]]:
        """
        Chia ảnh thành tiles dựa trên aspect ratio
        Returns: (list of tile images, (num_tiles_w, num_tiles_h))
        """
        width, height = image.size
        
        # Tìm aspect ratio tối ưu
        w_tiles, h_tiles = self.find_best_aspect_ratio(width, height)
        
        # Resize ảnh về kích thước grid
        target_w = w_tiles * self.image_size
        target_h = h_tiles * self.image_size
        resized = image.resize((target_w, target_h), Image.LANCZOS)
        
        # Chia thành tiles
        tiles = []
        for i in range(h_tiles):
            for j in range(w_tiles):
                left = j * self.image_size
                top = i * self.image_size
                right = left + self.image_size
                bottom = top + self.image_size
                tile = resized.crop((left, top, right, bottom))
                tiles.append(tile)
        
        return tiles, (w_tiles, h_tiles)
    
    def create_thumbnail(self, image: Image.Image) -> Image.Image:
        """Tạo thumbnail từ ảnh gốc"""
        return image.resize(
            (self.image_size, self.image_size), 
            Image.LANCZOS
        )
    
    def preprocess(
        self, 
        image: Image.Image,
        return_tensors: str = "pt"
    ) -> dict:
        """
        Preprocess ảnh: tạo tiles + thumbnail
        Returns: dict với pixel_values và metadata
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Lấy tiles
        tiles, (w_tiles, h_tiles) = self.get_tiles(image, self.max_tiles)
        
        # Tạo thumbnail
        thumbnail = self.create_thumbnail(image)
        
        # Kết hợp: thumbnail đầu tiên, sau đó là tiles
        all_images = [thumbnail] + tiles
        
        # Transform sang tensors
        pixel_values = torch.stack([
            self.transform(img) for img in all_images
        ])  # Shape: (1 + num_tiles, 3, H, W)
        
        return {
            "pixel_values": pixel_values,
            "num_tiles": len(tiles),
            "tile_grid": (w_tiles, h_tiles),
            "num_patches": len(all_images),  # thumbnail + tiles
        }
    
    def __call__(
        self, 
        images: List[Image.Image],
        return_tensors: str = "pt"
    ) -> dict:
        """Process batch of images"""
        batch_pixel_values = []
        batch_num_patches = []
        
        for image in images:
            result = self.preprocess(image, return_tensors)
            batch_pixel_values.append(result["pixel_values"])
            batch_num_patches.append(result["num_patches"])
        
        # Pad để có cùng số patches
        max_patches = max(batch_num_patches)
        padded_pixel_values = []
        
        for pv, num_patches in zip(batch_pixel_values, batch_num_patches):
            if num_patches < max_patches:
                # Pad với zeros
                pad_size = max_patches - num_patches
                pad = torch.zeros(
                    pad_size, 3, self.image_size, self.image_size,
                    dtype=pv.dtype
                )
                pv = torch.cat([pv, pad], dim=0)
            padded_pixel_values.append(pv)
        
        return {
            "pixel_values": torch.stack(padded_pixel_values),
            "num_patches": batch_num_patches,
        }
