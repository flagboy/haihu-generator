"""
チェックポイント管理器

モデルの保存・読み込み責務を分離し、
チェックポイントの管理を一元化する
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

from ....utils.logger import LoggerMixin


class CheckpointManager(LoggerMixin):
    """チェックポイント管理クラス"""

    def __init__(self, checkpoint_dir: str = "data/training/checkpoints"):
        """
        初期化

        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
        config: Any,
        session_id: str,
        is_best: bool = False,
        additional_info: dict[str, Any] | None = None,
    ) -> str | None:
        """
        チェックポイントを保存

        Args:
            model: 保存するモデル
            optimizer: オプティマイザー
            epoch: エポック番号
            metrics: メトリクス
            config: 訓練設定
            session_id: セッションID
            is_best: ベストモデルかどうか
            additional_info: 追加情報

        Returns:
            保存したファイルパス
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorchが利用できません")
            return None

        try:
            # セッション固有のディレクトリを作成
            session_dir = self.checkpoint_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            # ファイル名を決定
            if is_best:
                filename = "best_model.pt"
                # 以前のベストモデルをバックアップ
                self._backup_best_model(session_dir)
            else:
                filename = f"checkpoint_epoch_{epoch:04d}.pt"

            filepath = session_dir / filename

            # チェックポイントデータを構築
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": self._serialize_config(config),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "model_architecture": str(type(model).__name__),
            }

            # 追加情報があれば含める
            if additional_info:
                checkpoint["additional_info"] = additional_info

            # 保存
            torch.save(checkpoint, filepath)

            # メタデータを保存
            self._save_metadata(session_dir, epoch, metrics, is_best)

            # 古いチェックポイントをクリーンアップ
            if not is_best:
                self._cleanup_old_checkpoints(session_dir)

            self.logger.info(f"チェックポイント保存: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"チェックポイント保存に失敗: {e}")
            return None

    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        チェックポイントを読み込み

        Args:
            filepath: チェックポイントファイルパス
            model: 読み込み先のモデル
            optimizer: 読み込み先のオプティマイザー（オプション）
            strict: 厳密なパラメータマッチングを行うか

        Returns:
            チェックポイント情報
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorchが利用できません")
            return {}

        try:
            # デバイスを自動検出
            from ....utils.device_utils import get_available_device

            device = get_available_device() or torch.device("cpu")

            # チェックポイントを読み込み
            checkpoint = torch.load(filepath, map_location=device)

            # モデルの状態を復元
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            else:
                # 旧形式のチェックポイントに対応
                model.load_state_dict(checkpoint, strict=strict)
                return {"epoch": 0, "metrics": {}}

            # オプティマイザーの状態を復元
            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.logger.info(f"チェックポイント読み込み成功: {filepath}")

            return {
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
                "config": checkpoint.get("config", {}),
                "timestamp": checkpoint.get("timestamp", ""),
                "additional_info": checkpoint.get("additional_info", {}),
            }

        except Exception as e:
            self.logger.error(f"チェックポイント読み込みに失敗: {e}")
            return {}

    def find_best_checkpoint(self, session_id: str) -> str | None:
        """
        セッションのベストチェックポイントを探す

        Args:
            session_id: セッションID

        Returns:
            ベストチェックポイントのパス
        """
        session_dir = self.checkpoint_dir / session_id
        best_path = session_dir / "best_model.pt"

        if best_path.exists():
            return str(best_path)

        # ベストモデルがない場合は最新のチェックポイントを返す
        return self.find_latest_checkpoint(session_id)

    def find_latest_checkpoint(self, session_id: str) -> str | None:
        """
        セッションの最新チェックポイントを探す

        Args:
            session_id: セッションID

        Returns:
            最新チェックポイントのパス
        """
        session_dir = self.checkpoint_dir / session_id

        if not session_dir.exists():
            return None

        # チェックポイントファイルを探す
        checkpoint_files = list(session_dir.glob("checkpoint_epoch_*.pt"))

        if not checkpoint_files:
            # ベストモデルがあるか確認
            best_path = session_dir / "best_model.pt"
            if best_path.exists():
                return str(best_path)
            return None

        # エポック番号でソート
        checkpoint_files.sort(key=lambda p: int(p.stem.split("_")[-1]))

        return str(checkpoint_files[-1])

    def list_checkpoints(self, session_id: str) -> dict[str, Any]:
        """
        セッションのチェックポイント一覧を取得

        Args:
            session_id: セッションID

        Returns:
            チェックポイント情報
        """
        session_dir = self.checkpoint_dir / session_id

        if not session_dir.exists():
            return {"checkpoints": [], "best_model": None}

        checkpoints = []

        # 通常のチェックポイント
        for checkpoint_file in session_dir.glob("checkpoint_epoch_*.pt"):
            epoch = int(checkpoint_file.stem.split("_")[-1])
            checkpoints.append(
                {
                    "path": str(checkpoint_file),
                    "epoch": epoch,
                    "size_mb": checkpoint_file.stat().st_size / 1024 / 1024,
                    "modified": datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat(),
                }
            )

        # エポック順にソート
        checkpoints.sort(key=lambda x: x["epoch"])

        # ベストモデル
        best_path = session_dir / "best_model.pt"
        best_model = None
        if best_path.exists():
            best_model = {
                "path": str(best_path),
                "size_mb": best_path.stat().st_size / 1024 / 1024,
                "modified": datetime.fromtimestamp(best_path.stat().st_mtime).isoformat(),
            }

        # メタデータを読み込み
        metadata = self._load_metadata(session_dir)

        return {"checkpoints": checkpoints, "best_model": best_model, "metadata": metadata}

    def _serialize_config(self, config: Any) -> dict[str, Any]:
        """設定をシリアライズ"""
        if hasattr(config, "__dict__"):
            return config.__dict__
        elif isinstance(config, dict):
            return config
        else:
            return {"config": str(config)}

    def _backup_best_model(self, session_dir: Path):
        """ベストモデルをバックアップ"""
        best_path = session_dir / "best_model.pt"
        if best_path.exists():
            backup_path = (
                session_dir / f"best_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            shutil.copy2(best_path, backup_path)

    def _cleanup_old_checkpoints(self, session_dir: Path, keep_last: int = 5):
        """古いチェックポイントを削除"""
        checkpoint_files = list(session_dir.glob("checkpoint_epoch_*.pt"))

        if len(checkpoint_files) <= keep_last:
            return

        # エポック番号でソート
        checkpoint_files.sort(key=lambda p: int(p.stem.split("_")[-1]))

        # 古いファイルを削除
        for old_file in checkpoint_files[:-keep_last]:
            try:
                old_file.unlink()
                self.logger.info(f"古いチェックポイントを削除: {old_file}")
            except Exception as e:
                self.logger.warning(f"チェックポイント削除に失敗: {old_file}, エラー: {e}")

    def _save_metadata(
        self, session_dir: Path, epoch: int, metrics: dict[str, float], is_best: bool
    ):
        """メタデータを保存"""
        metadata_path = session_dir / "metadata.json"

        # 既存のメタデータを読み込み
        metadata = self._load_metadata(session_dir)

        # 更新
        metadata["last_epoch"] = epoch
        metadata["last_update"] = datetime.now().isoformat()

        if is_best:
            metadata["best_epoch"] = epoch
            metadata["best_metrics"] = metrics

        # 履歴に追加
        if "history" not in metadata:
            metadata["history"] = []

        metadata["history"].append(
            {
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "is_best": is_best,
            }
        )

        # 保存
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"メタデータ保存に失敗: {e}")

    def _load_metadata(self, session_dir: Path) -> dict[str, Any]:
        """メタデータを読み込み"""
        metadata_path = session_dir / "metadata.json"

        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path) as f:
                return json.load(f)
        except Exception:
            return {}
