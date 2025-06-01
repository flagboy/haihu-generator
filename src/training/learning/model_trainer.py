"""
モデル訓練システム

検出・分類モデルの訓練、データ拡張、GPU最適化を行う
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from ..annotation_data import AnnotationData, FrameAnnotation, TileAnnotation
from ...utils.logger import LoggerMixin
from ...utils.config import ConfigManager
from .learning_scheduler import LearningScheduler


class TileDataset(Dataset):
    """麻雀牌データセット"""
    
    def __init__(self, annotation_data: AnnotationData, model_type: str = "detection", 
                 transform=None, augment: bool = False):
        """
        初期化
        
        Args:
            annotation_data: アノテーションデータ
            model_type: モデルタイプ ("detection" or "classification")
            transform: 画像変換
            augment: データ拡張を使用するか
        """
        self.annotation_data = annotation_data
        self.model_type = model_type
        self.transform = transform
        self.augment = augment
        
        # データを準備
        self.samples = self._prepare_samples()
        
        # クラスマッピングを作成
        self.class_mapping = self._create_class_mapping()
        self.num_classes = len(self.class_mapping)
    
    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """サンプルデータを準備"""
        samples = []
        
        for video_annotation in self.annotation_data.video_annotations.values():
            for frame in video_annotation.frames:
                if not frame.is_valid or len(frame.tiles) == 0:
                    continue
                
                if self.model_type == "detection":
                    # 検出用: フレーム全体と全ての牌情報
                    samples.append({
                        "image_path": frame.image_path,
                        "image_width": frame.image_width,
                        "image_height": frame.image_height,
                        "tiles": frame.tiles,
                        "frame_id": frame.frame_id
                    })
                elif self.model_type == "classification":
                    # 分類用: 個別の牌画像
                    for tile in frame.tiles:
                        samples.append({
                            "image_path": frame.image_path,
                            "bbox": tile.bbox,
                            "tile_id": tile.tile_id,
                            "confidence": tile.confidence,
                            "frame_id": frame.frame_id
                        })
        
        return samples
    
    def _create_class_mapping(self) -> Dict[str, int]:
        """クラスマッピングを作成"""
        all_tile_ids = set()
        
        for video_annotation in self.annotation_data.video_annotations.values():
            for frame in video_annotation.frames:
                for tile in frame.tiles:
                    all_tile_ids.add(tile.tile_id)
        
        return {tile_id: idx for idx, tile_id in enumerate(sorted(all_tile_ids))}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # 画像を読み込み
            image = Image.open(sample["image_path"]).convert("RGB")
            
            if self.model_type == "detection":
                return self._get_detection_item(image, sample)
            elif self.model_type == "classification":
                return self._get_classification_item(image, sample)
            
        except Exception as e:
            # エラーの場合はダミーデータを返す
            if self.model_type == "detection":
                return torch.zeros(3, 416, 416), torch.zeros(5)  # ダミーターゲット
            else:
                return torch.zeros(3, 224, 224), torch.tensor(0, dtype=torch.long)
    
    def _get_detection_item(self, image: Image.Image, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """検出用のアイテムを取得"""
        # 画像をリサイズ
        target_size = (416, 416)
        image = image.resize(target_size)
        
        # 画像変換
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # ターゲット作成（簡易版）
        # 実際のYOLOでは複雑なターゲット形式が必要
        tiles = sample["tiles"]
        if tiles:
            # 最初の牌の情報を使用（簡易実装）
            tile = tiles[0]
            bbox = tile.bbox
            
            # 正規化されたバウンディングボックス
            x_center = (bbox.x1 + bbox.x2) / 2 / sample["image_width"]
            y_center = (bbox.y1 + bbox.y2) / 2 / sample["image_height"]
            width = bbox.width / sample["image_width"]
            height = bbox.height / sample["image_height"]
            
            class_id = self.class_mapping.get(tile.tile_id, 0)
            
            target = torch.tensor([x_center, y_center, width, height, class_id], dtype=torch.float32)
        else:
            target = torch.zeros(5, dtype=torch.float32)
        
        return image, target
    
    def _get_classification_item(self, image: Image.Image, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """分類用のアイテムを取得"""
        # バウンディングボックスで切り出し
        bbox = sample["bbox"]
        image = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        
        # リサイズ
        image = image.resize((224, 224))
        
        # データ拡張
        if self.augment and np.random.random() > 0.5:
            image = self._apply_augmentation(image)
        
        # 画像変換
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # ラベル
        class_id = self.class_mapping.get(sample["tile_id"], 0)
        target = torch.tensor(class_id, dtype=torch.long)
        
        return image, target
    
    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """データ拡張を適用"""
        # ランダム回転
        if np.random.random() > 0.7:
            angle = np.random.uniform(-15, 15)
            image = image.rotate(angle)
        
        # ランダム明度調整
        if np.random.random() > 0.7:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            factor = np.random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # ランダムコントラスト調整
        if np.random.random() > 0.7:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            factor = np.random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        return image


class ModelTrainer(LoggerMixin):
    """モデル訓練クラス"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()
        
        # 学習スケジューラー
        self.scheduler = LearningScheduler(config_manager)
        
        # 訓練状態管理
        self.training_sessions: Dict[str, Dict[str, Any]] = {}
        self.stop_flags: Dict[str, bool] = {}
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用デバイス: {self.device}")
    
    def train_model(self, model: nn.Module, train_data: AnnotationData, 
                   val_data: AnnotationData, config: Any, session_id: str) -> Dict[str, Any]:
        """
        モデルを訓練
        
        Args:
            model: 訓練するモデル
            train_data: 訓練データ
            val_data: 検証データ
            config: 訓練設定
            session_id: セッションID
            
        Returns:
            訓練結果
        """
        self.logger.info(f"モデル訓練開始: {config.model_type}, セッション: {session_id}")
        
        # 訓練状態を初期化
        self.training_sessions[session_id] = {
            "start_time": datetime.now(),
            "current_epoch": 0,
            "best_loss": float('inf'),
            "best_accuracy": 0.0,
            "training_history": [],
            "status": "running"
        }
        self.stop_flags[session_id] = False
        
        try:
            # モデルをデバイスに移動
            model = model.to(self.device)
            
            # データローダーを作成
            train_loader = self._create_dataloader(train_data, config, is_training=True)
            val_loader = self._create_dataloader(val_data, config, is_training=False)
            
            # 最適化器とスケジューラーを設定
            optimizer = self._create_optimizer(model, config)
            lr_scheduler = self._create_lr_scheduler(optimizer, config)
            
            # 損失関数を設定
            criterion = self._create_criterion(config)
            
            # 訓練ループ
            best_model_state = None
            best_model_path = None
            early_stopping_counter = 0
            
            for epoch in range(config.epochs):
                if self.stop_flags.get(session_id, False):
                    self.logger.info(f"訓練停止要求: セッション {session_id}")
                    break
                
                self.training_sessions[session_id]["current_epoch"] = epoch
                
                # 訓練フェーズ
                train_metrics = self._train_epoch(
                    model, train_loader, optimizer, criterion, epoch, session_id
                )
                
                # 検証フェーズ
                val_metrics = self._validate_epoch(
                    model, val_loader, criterion, epoch, session_id
                )
                
                # 学習率スケジューラー更新
                if lr_scheduler:
                    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(val_metrics['loss'])
                    else:
                        lr_scheduler.step()
                
                # 履歴を記録
                epoch_history = {
                    "epoch": epoch,
                    "train_loss": train_metrics['loss'],
                    "train_accuracy": train_metrics.get('accuracy', 0.0),
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics.get('accuracy', 0.0),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "timestamp": datetime.now().isoformat()
                }
                self.training_sessions[session_id]["training_history"].append(epoch_history)
                
                # ベストモデルの保存
                is_best = False
                if config.model_type == "classification":
                    if val_metrics.get('accuracy', 0.0) > self.training_sessions[session_id]["best_accuracy"]:
                        self.training_sessions[session_id]["best_accuracy"] = val_metrics['accuracy']
                        is_best = True
                else:
                    if val_metrics['loss'] < self.training_sessions[session_id]["best_loss"]:
                        self.training_sessions[session_id]["best_loss"] = val_metrics['loss']
                        is_best = True
                
                if is_best:
                    best_model_state = model.state_dict().copy()
                    best_model_path = self._save_checkpoint(
                        model, optimizer, epoch, val_metrics, config, session_id, is_best=True
                    )
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                # 定期的なチェックポイント保存
                if epoch % 10 == 0:
                    self._save_checkpoint(
                        model, optimizer, epoch, val_metrics, config, session_id, is_best=False
                    )
                
                # 早期停止チェック
                if early_stopping_counter >= config.early_stopping_patience:
                    self.logger.info(f"早期停止: {config.early_stopping_patience}エポック改善なし")
                    break
                
                self.logger.info(
                    f"エポック {epoch}: 訓練損失={train_metrics['loss']:.4f}, "
                    f"検証損失={val_metrics['loss']:.4f}, "
                    f"検証精度={val_metrics.get('accuracy', 0.0):.4f}"
                )
            
            # 訓練完了
            self.training_sessions[session_id]["status"] = "completed"
            
            # 最終メトリクス
            final_metrics = {
                "best_val_loss": self.training_sessions[session_id]["best_loss"],
                "best_val_accuracy": self.training_sessions[session_id]["best_accuracy"],
                "total_epochs": epoch + 1,
                "training_time": (datetime.now() - self.training_sessions[session_id]["start_time"]).total_seconds()
            }
            
            return {
                "best_model_path": best_model_path,
                "final_metrics": final_metrics,
                "training_history": self.training_sessions[session_id]["training_history"]
            }
            
        except Exception as e:
            self.training_sessions[session_id]["status"] = "failed"
            self.logger.error(f"モデル訓練に失敗: {e}")
            raise
        
        finally:
            # クリーンアップ
            if session_id in self.stop_flags:
                del self.stop_flags[session_id]
    
    def _create_dataloader(self, annotation_data: AnnotationData, config: Any, 
                          is_training: bool) -> DataLoader:
        """データローダーを作成"""
        # 画像変換を定義
        if is_training and config.use_data_augmentation:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # データセットを作成
        dataset = TileDataset(
            annotation_data=annotation_data,
            model_type=config.model_type,
            transform=transform,
            augment=is_training and config.use_data_augmentation
        )
        
        # データローダーを作成
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=is_training,
            num_workers=config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def _create_optimizer(self, model: nn.Module, config: Any) -> optim.Optimizer:
        """最適化器を作成"""
        if hasattr(config, 'optimizer_type'):
            optimizer_type = config.optimizer_type
        else:
            optimizer_type = "adam"
        
        if optimizer_type.lower() == "adam":
            return optim.Adam(model.parameters(), lr=config.learning_rate)
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(model.parameters(), lr=config.learning_rate)
        else:
            return optim.Adam(model.parameters(), lr=config.learning_rate)
    
    def _create_lr_scheduler(self, optimizer: optim.Optimizer, config: Any) -> Optional[Any]:
        """学習率スケジューラーを作成"""
        if hasattr(config, 'lr_scheduler_type'):
            scheduler_type = config.lr_scheduler_type
        else:
            scheduler_type = "plateau"
        
        if scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        else:
            return None
    
    def _create_criterion(self, config: Any) -> nn.Module:
        """損失関数を作成"""
        if config.model_type == "detection":
            # 検出用の損失関数（簡易版）
            return nn.MSELoss()  # バウンディングボックス回帰用
        elif config.model_type == "classification":
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss()
    
    def _train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module, 
                    epoch: int, session_id: str) -> Dict[str, float]:
        """1エポックの訓練"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if self.stop_flags.get(session_id, False):
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'forward') and 'SimpleCNN' in str(type(model)):
                # 検出モデルの場合
                bbox_pred, conf_pred, class_pred = model(data)
                
                # 簡易損失計算
                bbox_loss = criterion(bbox_pred, target[:, :4])
                conf_loss = nn.BCELoss()(conf_pred.squeeze(), (target[:, 4] > 0).float())
                class_loss = nn.CrossEntropyLoss()(class_pred, target[:, 4].long())
                
                loss = bbox_loss + conf_loss + class_loss
            else:
                # 分類モデルの場合
                output = model(data)
                loss = criterion(output, target)
                
                # 精度計算
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        metrics = {
            'loss': total_loss / len(dataloader)
        }
        
        if total > 0:
            metrics['accuracy'] = correct / total
        
        return metrics
    
    def _validate_epoch(self, model: nn.Module, dataloader: DataLoader, 
                       criterion: nn.Module, epoch: int, session_id: str) -> Dict[str, float]:
        """1エポックの検証"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                if self.stop_flags.get(session_id, False):
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                if hasattr(model, 'forward') and 'SimpleCNN' in str(type(model)):
                    # 検出モデルの場合
                    bbox_pred, conf_pred, class_pred = model(data)
                    
                    bbox_loss = criterion(bbox_pred, target[:, :4])
                    conf_loss = nn.BCELoss()(conf_pred.squeeze(), (target[:, 4] > 0).float())
                    class_loss = nn.CrossEntropyLoss()(class_pred, target[:, 4].long())
                    
                    loss = bbox_loss + conf_loss + class_loss
                else:
                    # 分類モデルの場合
                    output = model(data)
                    loss = criterion(output, target)
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                
                total_loss += loss.item()
        
        metrics = {
            'loss': total_loss / len(dataloader)
        }
        
        if total > 0:
            metrics['accuracy'] = correct / total
        
        return metrics
    
    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                        epoch: int, metrics: Dict[str, float], config: Any, 
                        session_id: str, is_best: bool = False) -> Optional[str]:
        """チェックポイントを保存"""
        try:
            checkpoint_dir = Path(self.config.get("training", {}).get("training_root", "data/training")) / "checkpoints" / session_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            if is_best:
                filename = "best_model.pt"
            else:
                filename = f"checkpoint_epoch_{epoch}.pt"
            
            filepath = checkpoint_dir / filename
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config.__dict__ if hasattr(config, '__dict__') else config,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, filepath)
            
            if is_best:
                self.logger.info(f"ベストモデル保存: {filepath}")
                return str(filepath)
            
            return None
            
        except Exception as e:
            self.logger.error(f"チェックポイント保存に失敗: {e}")
            return None
    
    def stop_training(self, session_id: str):
        """訓練を停止"""
        self.stop_flags[session_id] = True
        if session_id in self.training_sessions:
            self.training_sessions[session_id]["status"] = "stopping"
    
    def get_training_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """訓練進捗を取得"""
        return self.training_sessions.get(session_id)