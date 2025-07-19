from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import os
import torch
import torch.nn as nn
import glob

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenDINOV2ImageEmbedder(AbstractEncoder):
    """Uses the dinov2 transformer encoder for images"""
    def __init__(self, version='facebook/dinov2-base', freeze=True, device="cuda"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(version)
        self.model = AutoModel.from_pretrained(version)
        
        self.device = device
        self.model.to(device)
        
        if freeze: 
            self.freeze()
        
        # 修正拼写错误和属性名称
        print(f"{self.model.__class__.__name__} comes with {count_params(self.model) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, pixel_values):
        original_shape = pixel_values.shape
        if len(pixel_values.shape)==5:
            pixel_values = pixel_values.view(-1, *original_shape[2:])  # [B*T, C, H, W]

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state
        if len(original_shape)==5:
            last_hidden_state = last_hidden_state.view(
                original_shape[0], original_shape[1], last_hidden_state.shape[-2], last_hidden_state.shape[-1]
            )
        return last_hidden_state

    def forward(self, images):
        if not isinstance(images, list):
            images = [images]
        
        # 使用processor处理原始图像数据
        pixel_values = self.processor(images=images, return_tensors="pt").to(self.device)['pixel_values']
        return self.encode(pixel_values)

from transformers import CLIPModel, CLIPProcessor

class FrozenCLIPImageEmbedder(AbstractEncoder):
    """使用CLIP视觉编码器提取图像特征"""
    def __init__(self, version='openai/clip-vit-base-patch32', freeze=True, device="cuda"):
        super().__init__()
        # 加载CLIP的视觉处理组件
        self.processor = CLIPProcessor.from_pretrained(version).image_processor
        self.model = CLIPModel.from_pretrained(version).vision_model
        
        self.device = device
        self.model.to(device)
        
        if freeze:
            self.freeze()
        
        # 参数统计（保持与原实现一致）
        print(f"{self.model.__class__.__name__} comes with {count_params(self.model) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, pixel_values):
        # 维度处理（支持视频序列输入）
        original_shape = pixel_values.shape
        if len(original_shape) == 5:
            pixel_values = pixel_values.view(-1, *original_shape[2:])  # [B*T, C, H, W]

        # 特征提取
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        
        # CLIP特征选择：使用池化后的全局特征
        features = outputs.pooler_output  # [B*T, D]
        
        # 恢复时间维度（视频处理）
        if len(original_shape) == 5:
            features = features.view(original_shape[0], original_shape[1], -1)  # [B, T, D]
        
        return features

    def forward(self, images):
        # 输入预处理（适配CLIP官方处理方式）
        if not isinstance(images, list):
            images = [images]
        
        # CLIP官方预处理流程
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)
        
        return self.encode(inputs.pixel_values)


import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

class VideoFeatureExtractor:
    def __init__(self, model_path='facebook/dinov2-base', device='cuda'):
        self.device = device
        if 'dino' in model_path:
            self.embedder = FrozenDINOV2ImageEmbedder(
                version=model_path,
                freeze=True,
                device=device
            )
        elif 'clip' in model_path:
            self.embedder = FrozenCLIPImageEmbedder(
                version=model_path,
                freeze=True,
                device=device
            )
    
    def _frame_to_pil(self, frame):
        """将OpenCV BGR格式转换为PIL RGB格式"""
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def process_video(self, video_path, fps=4):
        """处理视频并提取特征"""
        # 初始化视频捕捉
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频属性
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / original_fps
        
        print(f"视频信息: {duration:.2f}秒, 原始FPS: {original_fps:.2f}")
        
        # 计算采样间隔（秒）
        interval = 1.0 / fps
        
        # 存储结果
        features = []
        sampled_frames = []
        current_time = 0.0
        
        with tqdm(total=int(duration * fps), desc="Processing video") as pbar:
            while cap.isOpened():
                # 设置当前时间戳
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为PIL图像
                pil_image = self._frame_to_pil(frame)
                sampled_frames.append(pil_image)
                
                # 每积累4帧处理一次（根据显存调整）
                if len(sampled_frames) >= 40:
                    # 提取特征
                    with torch.no_grad():
                        batch_features = self.embedder(sampled_frames)
                    features.append(batch_features.cpu())
                    sampled_frames = []
                
                # 更新时间戳
                current_time += interval
                pbar.update(1)
                
                # 超过视频时长时退出
                if current_time > duration:
                    break
        
        # 处理剩余帧
        if len(sampled_frames) > 0:
            with torch.no_grad():
                batch_features = self.embedder(sampled_frames)
            features.append(batch_features.cpu())
        
        # 释放资源
        cap.release()
        return torch.cat(features, dim=0) if features else None

# 使用示例
if __name__ == "__main__":
    # 初始化提取器
    # extractor = VideoFeatureExtractor(model_path='./useful_ckpt/dinov2',device="cuda")
    extractor = VideoFeatureExtractor(model_path='./useful_ckpt/clip-vit-base-patch32',device="cuda")
    video_root = "/home/guowenxiang/audio-visual-seld-dcase2023/data_dcase2023_task3/video_dev"
    video_paths = glob.glob(f'{video_root}/*/*.mp4')
    split = 3000
    num = 3
    for video_path in tqdm(sorted(video_paths)):  
        npy_path = video_path.replace('video', 'clip').replace('mp4', 'npy')
        if os.path.exists(npy_path):
            continue
        npy_dirname = os.path.dirname(npy_path)
        os.makedirs(npy_dirname, exist_ok=True)
        try:
            features = extractor.process_video(video_path).numpy()
            np.save(npy_path,features)
        except Exception as e:
            # 打印详细的错误信息
            print(f"\nError processing item {video_path} ")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            continue