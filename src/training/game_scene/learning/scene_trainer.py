"""
対局画面分類モデルの学習

データセットを使用してモデルを学習
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ....utils.logger import LoggerMixin
from ..core.game_scene_classifier import GameSceneClassifierModel
from .scene_dataset import SceneDataset


class SceneTrainer(LoggerMixin):
    """対局画面分類モデルのトレーナー"""

    def __init__(
        self, output_dir: str = "models/game_scene", device: str | None = None, num_workers: int = 4
    ):
        """
        初期化

        Args:
            output_dir: モデル保存ディレクトリ
            device: 使用デバイス
            num_workers: データローダーのワーカー数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers

        # デバイス設定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.logger.info(f"SceneTrainer初期化完了 (device: {self.device})")

    def train(
        self,
        train_dataset: SceneDataset,
        val_dataset: SceneDataset,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        checkpoint_interval: int = 5,
    ) -> dict[str, any]:
        """
        モデルを学習

        Args:
            train_dataset: 学習データセット
            val_dataset: 検証データセット
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            early_stopping_patience: 早期終了の待機エポック数
            checkpoint_interval: チェックポイント保存間隔

        Returns:
            学習結果
        """
        # セッションIDを生成
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # データローダーを作成
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        # モデルを初期化
        model = GameSceneClassifierModel(num_classes=2, pretrained=True)
        model.to(self.device)

        # 損失関数（クラスの重みを考慮）
        class_weights = train_dataset.get_class_weights().to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # オプティマイザー
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 学習率スケジューラー
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # 学習履歴
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        # 早期終了の設定
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = None

        self.logger.info(
            f"学習開始: {epochs}エポック, バッチサイズ={batch_size}, 学習率={learning_rate}"
        )

        # 学習ループ
        for epoch in range(epochs):
            # 学習
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)

            # 検証
            val_loss, val_acc = self._validate(model, val_loader, criterion)

            # 履歴を記録
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])

            # ログ出力
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )

            # 学習率を調整
            scheduler.step(val_loss)

            # モデル保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # ベストモデルを保存
                if best_model_path and Path(best_model_path).exists():
                    os.remove(best_model_path)

                best_model_path = session_dir / f"best_model_epoch{epoch + 1}.pth"
                torch.save(model.state_dict(), best_model_path)
                self.logger.info(f"ベストモデルを保存: {best_model_path}")
            else:
                patience_counter += 1

            # チェックポイント保存
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = session_dir / f"checkpoint_epoch{epoch + 1}.pth"
                self._save_checkpoint(model, optimizer, scheduler, epoch, history, checkpoint_path)

            # 早期終了
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"早期終了: {epoch + 1}エポック")
                break

        # 最終モデルを保存
        final_model_path = session_dir / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)

        # 学習結果を保存
        results = {
            "session_id": session_id,
            "epochs_trained": len(history["train_loss"]),
            "best_val_loss": best_val_loss,
            "best_val_acc": max(history["val_acc"]),
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
            "history": history,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "early_stopping_patience": early_stopping_patience,
                "device": str(self.device),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
            },
            "paths": {"best_model": str(best_model_path), "final_model": str(final_model_path)},
        }

        # 結果をJSONで保存
        results_path = session_dir / "training_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"学習完了: 結果を {session_dir} に保存")

        return results

    def _train_epoch(
        self, model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer
    ) -> tuple[float, float]:
        """
        1エポックの学習

        Args:
            model: モデル
            loader: データローダー
            criterion: 損失関数
            optimizer: オプティマイザー

        Returns:
            (平均損失, 精度)
        """
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        with tqdm(loader, desc="Training") as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 順伝播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 逆伝播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 統計を更新
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # プログレスバー更新
                pbar.set_postfix(
                    {"loss": total_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
                )

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate(
        self, model: nn.Module, loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        """
        検証

        Args:
            model: モデル
            loader: データローダー
            criterion: 損失関数

        Returns:
            (平均損失, 精度)
        """
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad(), tqdm(loader, desc="Validation") as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 順伝播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 統計を更新
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # プログレスバー更新
                pbar.set_postfix(
                    {"loss": total_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
                )

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: any,
        epoch: int,
        history: dict,
        path: Path,
    ):
        """チェックポイントを保存"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"チェックポイントを保存: {path}")

    def resume_training(
        self,
        checkpoint_path: str,
        train_dataset: SceneDataset,
        val_dataset: SceneDataset,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> dict[str, any]:
        """
        チェックポイントから学習を再開

        Args:
            checkpoint_path: チェックポイントファイルパス
            train_dataset: 学習データセット
            val_dataset: 検証データセット
            epochs: 追加エポック数
            batch_size: バッチサイズ

        Returns:
            学習結果
        """
        # チェックポイントを読み込み
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # モデルを復元
        model = GameSceneClassifierModel(num_classes=2, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        # オプティマイザーを復元
        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # スケジューラーを復元
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 履歴を復元
        self.history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1

        self.logger.info(f"チェックポイントから再開: エポック {start_epoch} から")

        # 学習を継続（実装は省略）
        # TODO: 実際の継続学習の実装

        return {}
