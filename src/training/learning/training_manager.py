"""
学習管理システム

学習プロセス全体の管理、継続学習、モデルバージョン管理を行う
"""

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ...models.model_manager import ModelManager
from ...utils.config import ConfigManager
from ...utils.logger import LoggerMixin
from ..annotation_data import AnnotationData
from ..dataset_manager import DatasetManager
from .learning_scheduler import LearningScheduler
from .model_evaluator import ModelEvaluator
from .model_trainer import ModelTrainer


@dataclass
class TrainingConfig:
    """学習設定"""

    model_type: str  # "detection" or "classification"
    model_name: str
    dataset_version_id: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 10
    save_best_only: bool = True
    use_data_augmentation: bool = True
    transfer_learning: bool = False
    pretrained_model_path: str | None = None
    gpu_enabled: bool = True
    num_workers: int = 4
    seed: int = 42


@dataclass
class TrainingSession:
    """学習セッション情報"""

    session_id: str
    config: TrainingConfig
    start_time: datetime
    end_time: datetime | None = None
    status: str = "running"  # "running", "completed", "failed", "stopped"
    best_model_path: str | None = None
    final_metrics: dict[str, float] | None = None
    training_history: list[dict[str, Any]] = None
    notes: str = ""


class TrainingManager(LoggerMixin):
    """学習管理クラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # 各種マネージャーの初期化
        self.dataset_manager = DatasetManager(config_manager)
        self.model_manager = ModelManager(config_manager)
        self.model_trainer = ModelTrainer(config_manager)
        self.learning_scheduler = LearningScheduler(config_manager)
        self.model_evaluator = ModelEvaluator(config_manager)

        # 学習関連ディレクトリ設定
        self.training_root = Path(
            self.config.get("training", {}).get("training_root", "data/training")
        )
        self.sessions_dir = self.training_root / "sessions"
        self.experiments_dir = self.training_root / "experiments"
        self.checkpoints_dir = self.training_root / "checkpoints"

        for dir_path in [self.sessions_dir, self.experiments_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # セッション管理
        self.sessions_file = self.sessions_dir / "sessions.json"
        self.active_sessions: dict[str, TrainingSession] = {}
        self._load_sessions()

        self.logger.info(f"TrainingManager初期化完了: {self.training_root}")

    def _load_sessions(self):
        """セッション情報を読み込み"""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, encoding="utf-8") as f:
                    sessions_data = json.load(f)

                for session_data in sessions_data.get("sessions", []):
                    session = TrainingSession(
                        session_id=session_data["session_id"],
                        config=TrainingConfig(**session_data["config"]),
                        start_time=datetime.fromisoformat(session_data["start_time"]),
                        end_time=datetime.fromisoformat(session_data["end_time"])
                        if session_data.get("end_time")
                        else None,
                        status=session_data.get("status", "unknown"),
                        best_model_path=session_data.get("best_model_path"),
                        final_metrics=session_data.get("final_metrics"),
                        training_history=session_data.get("training_history", []),
                        notes=session_data.get("notes", ""),
                    )
                    self.active_sessions[session.session_id] = session

            except Exception as e:
                self.logger.warning(f"セッション情報の読み込みに失敗: {e}")

    def _save_sessions(self):
        """セッション情報を保存"""
        try:
            sessions_data = {"sessions": [], "last_updated": datetime.now().isoformat()}

            for session in self.active_sessions.values():
                session_dict = asdict(session)
                session_dict["start_time"] = session.start_time.isoformat()
                if session.end_time:
                    session_dict["end_time"] = session.end_time.isoformat()
                sessions_data["sessions"].append(session_dict)

            with open(self.sessions_file, "w", encoding="utf-8") as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"セッション情報の保存に失敗: {e}")

    def start_training(self, training_config: TrainingConfig) -> str:
        """
        学習を開始

        Args:
            training_config: 学習設定

        Returns:
            セッションID
        """
        session_id = str(uuid.uuid4())
        session = TrainingSession(
            session_id=session_id, config=training_config, start_time=datetime.now()
        )

        self.active_sessions[session_id] = session
        self._save_sessions()

        try:
            self.logger.info(f"学習開始: セッション {session_id}")

            # データセット準備
            annotation_data = self._prepare_dataset(training_config.dataset_version_id)
            if not annotation_data:
                raise ValueError(f"データセットの準備に失敗: {training_config.dataset_version_id}")

            # データ分割
            train_data, val_data, test_data = self._split_dataset(
                annotation_data,
                training_config.validation_split,
                training_config.test_split,
                training_config.seed,
            )

            # モデル準備
            model = self._prepare_model(training_config)

            # 学習実行
            training_result = self.model_trainer.train_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                config=training_config,
                session_id=session_id,
            )

            # 結果保存
            session.status = "completed"
            session.end_time = datetime.now()
            session.best_model_path = training_result.get("best_model_path")
            session.final_metrics = training_result.get("final_metrics")
            session.training_history = training_result.get("training_history", [])

            # 最終評価
            if test_data and session.best_model_path:
                test_metrics = self.model_evaluator.evaluate_model(
                    model_path=session.best_model_path,
                    test_data=test_data,
                    model_type=training_config.model_type,
                )
                session.final_metrics.update({"test_" + k: v for k, v in test_metrics.items()})

            self._save_sessions()
            self.logger.info(f"学習完了: セッション {session_id}")

            return session_id

        except Exception as e:
            session.status = "failed"
            session.end_time = datetime.now()
            session.notes = f"エラー: {str(e)}"
            self._save_sessions()
            self.logger.error(f"学習失敗: セッション {session_id}, エラー: {e}")
            raise

    def _prepare_dataset(self, dataset_version_id: str) -> AnnotationData | None:
        """データセットを準備"""
        try:
            # バージョン情報を取得
            versions = self.dataset_manager.list_versions()
            version_info = None
            for version in versions:
                if version["id"] == dataset_version_id:
                    version_info = version
                    break

            if not version_info:
                self.logger.error(f"データセットバージョンが見つかりません: {dataset_version_id}")
                return None

            # アノテーションデータを読み込み
            version_dir = Path(version_info["export_path"])
            annotation_path = version_dir / "annotations.json"

            if not annotation_path.exists():
                self.logger.error(f"アノテーションファイルが見つかりません: {annotation_path}")
                return None

            annotation_data = AnnotationData()
            annotation_data.load_from_json(str(annotation_path))

            self.logger.info(f"データセット準備完了: {len(annotation_data.video_annotations)}動画")
            return annotation_data

        except Exception as e:
            self.logger.error(f"データセット準備に失敗: {e}")
            return None

    def _split_dataset(
        self, annotation_data: AnnotationData, val_split: float, test_split: float, seed: int
    ) -> tuple[AnnotationData, AnnotationData, AnnotationData]:
        """データセットを訓練/検証/テストに分割"""
        import random

        random.seed(seed)

        # 全フレームを収集
        all_frames = []
        for video_annotation in annotation_data.video_annotations.values():
            for frame in video_annotation.frames:
                if frame.is_valid and len(frame.tiles) > 0:
                    all_frames.append((video_annotation.video_id, frame))

        # シャッフル
        random.shuffle(all_frames)

        # 分割
        total_frames = len(all_frames)
        test_size = int(total_frames * test_split)
        val_size = int(total_frames * val_split)
        train_size = total_frames - test_size - val_size

        train_frames = all_frames[:train_size]
        val_frames = all_frames[train_size : train_size + val_size]
        test_frames = all_frames[train_size + val_size :]

        # AnnotationDataオブジェクトを作成
        train_data = self._create_annotation_data_from_frames(annotation_data, train_frames)
        val_data = self._create_annotation_data_from_frames(annotation_data, val_frames)
        test_data = self._create_annotation_data_from_frames(annotation_data, test_frames)

        self.logger.info(
            f"データ分割完了: 訓練={len(train_frames)}, 検証={len(val_frames)}, テスト={len(test_frames)}"
        )

        return train_data, val_data, test_data

    def _create_annotation_data_from_frames(
        self, original_data: AnnotationData, frame_list: list[tuple[str, Any]]
    ) -> AnnotationData:
        """フレームリストからAnnotationDataを作成"""
        new_data = AnnotationData()

        # 動画IDごとにフレームをグループ化
        video_frames = {}
        for video_id, frame in frame_list:
            if video_id not in video_frames:
                video_frames[video_id] = []
            video_frames[video_id].append(frame)

        # VideoAnnotationを作成
        for video_id, frames in video_frames.items():
            original_video = original_data.video_annotations[video_id]

            # 新しいVideoAnnotationを作成（フレームのみ異なる）
            from ..annotation_data import VideoAnnotation

            new_video = VideoAnnotation(
                video_id=original_video.video_id,
                video_path=original_video.video_path,
                video_name=original_video.video_name,
                duration=original_video.duration,
                fps=original_video.fps,
                width=original_video.width,
                height=original_video.height,
                frames=frames,
                created_at=original_video.created_at,
                updated_at=original_video.updated_at,
                metadata=original_video.metadata,
            )

            new_data.video_annotations[video_id] = new_video

        return new_data

    def _prepare_model(self, config: TrainingConfig) -> nn.Module:
        """モデルを準備"""
        if config.model_type == "detection":
            from ...detection.tile_detector import SimpleCNN

            # 牌の種類数を設定（設定ファイルから取得）
            num_classes = self.config.get("training", {}).get("num_tile_classes", 34)
            model = SimpleCNN(num_classes=num_classes)
        elif config.model_type == "classification":
            from ...classification.tile_classifier import TileClassifier

            model = TileClassifier(self.config_manager)
            model = model.model  # 内部のモデルを取得
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {config.model_type}")

        # 転移学習の場合
        if config.transfer_learning and config.pretrained_model_path:
            self.logger.info(f"事前学習済みモデルを読み込み: {config.pretrained_model_path}")
            if os.path.exists(config.pretrained_model_path):
                checkpoint = torch.load(config.pretrained_model_path, map_location="cpu")
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            else:
                self.logger.warning(
                    f"事前学習済みモデルが見つかりません: {config.pretrained_model_path}"
                )

        return model

    def stop_training(self, session_id: str) -> bool:
        """
        学習を停止

        Args:
            session_id: セッションID

        Returns:
            停止成功かどうか
        """
        if session_id not in self.active_sessions:
            self.logger.error(f"セッションが見つかりません: {session_id}")
            return False

        session = self.active_sessions[session_id]
        if session.status != "running":
            self.logger.warning(f"セッションは実行中ではありません: {session_id}")
            return False

        try:
            # トレーナーに停止を指示
            self.model_trainer.stop_training(session_id)

            session.status = "stopped"
            session.end_time = datetime.now()
            session.notes = "ユーザーによって停止されました"

            self._save_sessions()
            self.logger.info(f"学習停止: セッション {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"学習停止に失敗: {session_id}, エラー: {e}")
            return False

    def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        """
        セッション状態を取得

        Args:
            session_id: セッションID

        Returns:
            セッション情報
        """
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        # 実行中の場合は最新の進捗を取得
        current_progress = None
        if session.status == "running":
            current_progress = self.model_trainer.get_training_progress(session_id)

        return {
            "session_id": session.session_id,
            "config": asdict(session.config),
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "status": session.status,
            "best_model_path": session.best_model_path,
            "final_metrics": session.final_metrics,
            "training_history": session.training_history,
            "current_progress": current_progress,
            "notes": session.notes,
        }

    def list_sessions(self, status_filter: str | None = None) -> list[dict[str, Any]]:
        """
        セッション一覧を取得

        Args:
            status_filter: ステータスフィルター

        Returns:
            セッション一覧
        """
        sessions = []
        for session in self.active_sessions.values():
            if status_filter is None or session.status == status_filter:
                sessions.append(self.get_session_status(session.session_id))

        # 開始時間でソート（新しい順）
        sessions.sort(key=lambda x: x["start_time"], reverse=True)
        return sessions

    def continue_training(self, base_session_id: str, new_config: TrainingConfig) -> str:
        """
        継続学習を実行

        Args:
            base_session_id: ベースとなるセッションID
            new_config: 新しい学習設定

        Returns:
            新しいセッションID
        """
        base_session = self.active_sessions.get(base_session_id)
        if not base_session or not base_session.best_model_path:
            raise ValueError(f"ベースセッションまたはモデルが見つかりません: {base_session_id}")

        # 継続学習用の設定を更新
        new_config.transfer_learning = True
        new_config.pretrained_model_path = base_session.best_model_path

        self.logger.info(f"継続学習開始: ベース={base_session_id}")
        return self.start_training(new_config)

    def compare_models(self, session_ids: list[str]) -> dict[str, Any]:
        """
        複数のモデルを比較

        Args:
            session_ids: 比較するセッションIDのリスト

        Returns:
            比較結果
        """
        comparison_data = {"sessions": [], "metrics_comparison": {}, "best_session": None}

        valid_sessions = []
        for session_id in session_ids:
            session = self.active_sessions.get(session_id)
            if session and session.final_metrics:
                valid_sessions.append(session)
                comparison_data["sessions"].append(
                    {
                        "session_id": session.session_id,
                        "config": asdict(session.config),
                        "metrics": session.final_metrics,
                        "training_time": (session.end_time - session.start_time).total_seconds()
                        if session.end_time
                        else None,
                    }
                )

        if not valid_sessions:
            return comparison_data

        # メトリクス比較
        all_metrics = set()
        for session in valid_sessions:
            all_metrics.update(session.final_metrics.keys())

        for metric in all_metrics:
            values = []
            for session in valid_sessions:
                if metric in session.final_metrics:
                    values.append(
                        {"session_id": session.session_id, "value": session.final_metrics[metric]}
                    )

            if values:
                comparison_data["metrics_comparison"][metric] = {
                    "values": values,
                    "best": max(values, key=lambda x: x["value"]),
                    "worst": min(values, key=lambda x: x["value"]),
                }

        # 最良セッションを決定（検証精度ベース）
        if "val_accuracy" in comparison_data["metrics_comparison"]:
            best_session_id = comparison_data["metrics_comparison"]["val_accuracy"]["best"][
                "session_id"
            ]
            comparison_data["best_session"] = best_session_id

        return comparison_data

    def cleanup_old_sessions(self, keep_count: int = 10) -> bool:
        """
        古いセッションをクリーンアップ

        Args:
            keep_count: 保持するセッション数

        Returns:
            クリーンアップ成功かどうか
        """
        try:
            # 完了済みセッションを取得
            completed_sessions = [
                s for s in self.active_sessions.values() if s.status == "completed"
            ]

            if len(completed_sessions) <= keep_count:
                return True

            # 古い順にソート
            completed_sessions.sort(key=lambda x: x.start_time)

            # 削除対象を決定
            sessions_to_remove = completed_sessions[:-keep_count]

            for session in sessions_to_remove:
                # モデルファイルを削除
                if session.best_model_path and os.path.exists(session.best_model_path):
                    os.remove(session.best_model_path)

                # セッションディレクトリを削除
                session_dir = self.sessions_dir / session.session_id
                if session_dir.exists():
                    shutil.rmtree(session_dir)

                # メモリから削除
                del self.active_sessions[session.session_id]

            self._save_sessions()
            self.logger.info(f"古いセッションをクリーンアップ: {len(sessions_to_remove)}個削除")
            return True

        except Exception as e:
            self.logger.error(f"セッションクリーンアップに失敗: {e}")
            return False
