"""
モデル管理モジュール
学習済みモデルの保存・読み込み・管理を行う
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class ModelManager:
    """モデル管理クラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config = config_manager
        self.logger = get_logger(__name__)

        # モデルディレクトリ設定
        self.models_dir = self.config.get_config().get("directories", {}).get("models", "models")
        self.ensure_models_directory()

        # メタデータファイル
        self.metadata_file = os.path.join(self.models_dir, "models_metadata.json")
        self.metadata = self._load_metadata()

        self.logger.info(f"ModelManager initialized with models directory: {self.models_dir}")

    def ensure_models_directory(self):
        """モデルディレクトリの存在を確認・作成"""
        os.makedirs(self.models_dir, exist_ok=True)

        # サブディレクトリも作成
        subdirs = ["detection", "classification", "checkpoints", "exports"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.models_dir, subdir), exist_ok=True)

    def _load_metadata(self) -> dict[str, Any]:
        """メタデータファイルを読み込み"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")

        return {"models": {}}

    def _save_metadata(self):
        """メタデータファイルを保存"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def save_model(
        self,
        model: nn.Module,
        model_name: str,
        model_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        モデルを保存

        Args:
            model: 保存するモデル
            model_name: モデル名
            model_type: モデルタイプ（detection, classification）
            metadata: 追加メタデータ

        Returns:
            保存されたモデルファイルのパス
        """
        try:
            # ファイルパス生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.pt"
            model_path = os.path.join(self.models_dir, model_type, filename)

            # モデル保存
            save_dict = {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "model_type": model_type,
                "timestamp": timestamp,
                "pytorch_version": torch.__version__,
            }

            if metadata:
                save_dict.update(metadata)

            torch.save(save_dict, model_path)

            # メタデータ更新
            model_id = self._generate_model_id(model_name, timestamp)
            self.metadata["models"][model_id] = {
                "name": model_name,
                "type": model_type,
                "path": model_path,
                "timestamp": timestamp,
                "file_size": os.path.getsize(model_path),
                "checksum": self._calculate_checksum(model_path),
                "metadata": metadata or {},
            }

            self._save_metadata()

            self.logger.info(f"Model saved: {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {e}")
            raise

    def load_model(
        self, model: nn.Module, model_path: str, device: torch.device | None = None
    ) -> bool:
        """
        モデルを読み込み

        Args:
            model: 読み込み先のモデル
            model_path: モデルファイルのパス
            device: デバイス

        Returns:
            読み込み成功フラグ
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False

            # デバイス設定
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # モデル読み込み
            checkpoint = torch.load(model_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                self.logger.info(f"Model loaded from checkpoint: {model_path}")
            else:
                model.load_state_dict(checkpoint)
                self.logger.info(f"Model loaded: {model_path}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            return False

    def get_latest_model(self, model_type: str, model_name: str | None = None) -> str | None:
        """
        最新のモデルパスを取得

        Args:
            model_type: モデルタイプ
            model_name: モデル名（オプション）

        Returns:
            最新モデルのパス
        """
        matching_models = []

        for _model_id, model_info in self.metadata["models"].items():
            if model_info["type"] == model_type and (
                model_name is None or model_info["name"] == model_name
            ):
                matching_models.append(model_info)

        if not matching_models:
            return None

        # タイムスタンプでソートして最新を取得
        latest_model = max(matching_models, key=lambda x: x["timestamp"])
        return latest_model["path"]

    def list_models(self, model_type: str | None = None) -> list[dict[str, Any]]:
        """
        モデル一覧を取得

        Args:
            model_type: モデルタイプでフィルタ（オプション）

        Returns:
            モデル情報のリスト
        """
        models = []

        for model_id, model_info in self.metadata["models"].items():
            if model_type is None or model_info["type"] == model_type:
                models.append({"id": model_id, **model_info})

        # タイムスタンプでソート（新しい順）
        models.sort(key=lambda x: x["timestamp"], reverse=True)
        return models

    def delete_model(self, model_id: str) -> bool:
        """
        モデルを削除

        Args:
            model_id: モデルID

        Returns:
            削除成功フラグ
        """
        try:
            if model_id not in self.metadata["models"]:
                self.logger.warning(f"Model ID not found: {model_id}")
                return False

            model_info = self.metadata["models"][model_id]
            model_path = model_info["path"]

            # ファイル削除
            if os.path.exists(model_path):
                os.remove(model_path)
                self.logger.info(f"Model file deleted: {model_path}")

            # メタデータから削除
            del self.metadata["models"][model_id]
            self._save_metadata()

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def create_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        model_name: str,
    ) -> str:
        """
        学習チェックポイントを作成

        Args:
            model: モデル
            optimizer: オプティマイザー
            epoch: エポック数
            loss: 損失値
            model_name: モデル名

        Returns:
            チェックポイントファイルのパス
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_checkpoint_epoch_{epoch}_{timestamp}.pt"
            checkpoint_path = os.path.join(self.models_dir, "checkpoints", filename)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "model_name": model_name,
                "timestamp": timestamp,
            }

            torch.save(checkpoint, checkpoint_path)

            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise

    def load_checkpoint(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str
    ) -> dict[str, Any]:
        """
        チェックポイントを読み込み

        Args:
            model: モデル
            optimizer: オプティマイザー
            checkpoint_path: チェックポイントファイルのパス

        Returns:
            チェックポイント情報
        """
        try:
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")

            return {
                "epoch": checkpoint.get("epoch", 0),
                "loss": checkpoint.get("loss", 0.0),
                "model_name": checkpoint.get("model_name", ""),
                "timestamp": checkpoint.get("timestamp", ""),
            }

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def export_model(self, model: nn.Module, model_name: str, export_format: str = "onnx") -> str:
        """
        モデルを他の形式でエクスポート

        Args:
            model: エクスポートするモデル
            model_name: モデル名
            export_format: エクスポート形式（onnx, torchscript）

        Returns:
            エクスポートファイルのパス
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if export_format == "onnx":
                filename = f"{model_name}_{timestamp}.onnx"
                export_path = os.path.join(self.models_dir, "exports", filename)

                # ダミー入力を作成（実際の入力サイズに合わせて調整が必要）
                dummy_input = torch.randn(1, 3, 224, 224)

                torch.onnx.export(
                    model,
                    dummy_input,
                    export_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                )

            elif export_format == "torchscript":
                filename = f"{model_name}_{timestamp}.pt"
                export_path = os.path.join(self.models_dir, "exports", filename)

                scripted_model = torch.jit.script(model)
                scripted_model.save(export_path)

            else:
                raise ValueError(f"Unsupported export format: {export_format}")

            self.logger.info(f"Model exported to {export_format}: {export_path}")
            return export_path

        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            raise

    def optimize_model(
        self, model: nn.Module, optimization_type: str = "quantization"
    ) -> nn.Module:
        """
        モデルを最適化

        Args:
            model: 最適化するモデル
            optimization_type: 最適化タイプ

        Returns:
            最適化されたモデル
        """
        try:
            if optimization_type == "quantization":
                # 動的量子化
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                self.logger.info("Model quantized successfully")
                return quantized_model

            elif optimization_type == "pruning":
                # 構造化プルーニング（簡易版）
                import torch.nn.utils.prune as prune

                for module in model.modules():
                    if isinstance(module, nn.Conv2d | nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=0.2)

                self.logger.info("Model pruned successfully")
                return model

            else:
                raise ValueError(f"Unsupported optimization type: {optimization_type}")

        except Exception as e:
            self.logger.error(f"Failed to optimize model: {e}")
            raise

    def _generate_model_id(self, model_name: str, timestamp: str) -> str:
        """モデルIDを生成"""
        return f"{model_name}_{timestamp}"

    def _calculate_checksum(self, file_path: str) -> str:
        """ファイルのチェックサムを計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def cleanup_old_models(self, keep_count: int = 5, model_type: str | None = None):
        """古いモデルをクリーンアップ"""
        try:
            models = self.list_models(model_type)

            if len(models) <= keep_count:
                return

            # 古いモデルを削除
            models_to_delete = models[keep_count:]

            for model_info in models_to_delete:
                self.delete_model(model_info["id"])

            self.logger.info(f"Cleaned up {len(models_to_delete)} old models")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old models: {e}")

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """モデル情報を取得"""
        return self.metadata["models"].get(model_id)

    def verify_model_integrity(self, model_id: str) -> bool:
        """モデルファイルの整合性を検証"""
        try:
            model_info = self.get_model_info(model_id)
            if not model_info:
                return False

            model_path = model_info["path"]
            if not os.path.exists(model_path):
                return False

            # ファイルサイズチェック
            current_size = os.path.getsize(model_path)
            if current_size != model_info["file_size"]:
                return False

            # チェックサムチェック
            current_checksum = self._calculate_checksum(model_path)
            return current_checksum == model_info["checksum"]

        except Exception as e:
            self.logger.error(f"Failed to verify model integrity: {e}")
            return False
