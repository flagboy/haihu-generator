"""
継続学習コントローラー

新しいデータが追加された際の自動学習、知識の保持、
学習戦略の管理を行う
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ....utils.logger import LoggerMixin
from ...annotation_data import AnnotationData
from ..dataset_manager import DatasetManager
from .components.checkpoint_manager import CheckpointManager
from .training_manager import TrainingConfig, TrainingManager


@dataclass
class ContinuousLearningConfig:
    """継続学習設定"""

    # 基本設定
    base_model_path: str | None = None
    incremental_data_threshold: int = 100  # 新しいデータが何件たまったら学習するか

    # 学習戦略
    strategy: Literal["fine_tuning", "elastic_weight", "rehearsal", "ewc"] = "fine_tuning"

    # Fine-tuning設定
    fine_tuning_lr_factor: float = 0.1  # ベース学習率に対する倍率
    freeze_layers: list[str] = field(default_factory=list)  # 凍結するレイヤー名

    # Rehearsal設定（過去データの再生）
    rehearsal_size: int = 1000  # 保持する過去データ数
    rehearsal_ratio: float = 0.3  # バッチ内の過去データの割合

    # EWC（Elastic Weight Consolidation）設定
    ewc_lambda: float = 0.5  # EWCの重み
    fisher_samples: int = 200  # Fisher情報行列計算用のサンプル数

    # 知識蒸留設定
    use_knowledge_distillation: bool = False
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7

    # 自動学習設定
    auto_train_enabled: bool = True
    check_interval_hours: int = 24
    min_performance_threshold: float = 0.85  # 最低性能閾値

    # データ管理
    data_versioning: bool = True
    max_data_versions: int = 5


@dataclass
class ContinuousLearningSession:
    """継続学習セッション"""

    session_id: str
    base_session_id: str | None
    config: ContinuousLearningConfig
    start_time: datetime
    last_update_time: datetime
    total_samples_seen: int = 0
    incremental_updates: int = 0
    performance_history: list[dict[str, float]] = field(default_factory=list)
    data_versions: list[str] = field(default_factory=list)


class ContinuousLearningController(LoggerMixin):
    """継続学習コントローラークラス"""

    def __init__(
        self,
        training_manager: TrainingManager,
        dataset_manager: DatasetManager,
        checkpoint_manager: CheckpointManager,
    ):
        """
        初期化

        Args:
            training_manager: 学習管理器
            dataset_manager: データセット管理器
            checkpoint_manager: チェックポイント管理器
        """
        self.training_manager = training_manager
        self.dataset_manager = dataset_manager
        self.checkpoint_manager = checkpoint_manager

        # セッション管理
        self.sessions_file = Path("data/training/continuous_sessions.json")
        self.active_sessions: dict[str, ContinuousLearningSession] = {}
        self._load_sessions()

        # リハーサルバッファ（過去データ保存用）
        self.rehearsal_buffer = {}

        # Fisher情報行列キャッシュ（EWC用）
        self.fisher_cache = {}

    def _load_sessions(self):
        """セッション情報を読み込み"""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file) as f:
                    data = json.load(f)
                    for session_data in data.get("sessions", []):
                        # セッションを復元
                        config = ContinuousLearningConfig(**session_data["config"])
                        session = ContinuousLearningSession(
                            session_id=session_data["session_id"],
                            base_session_id=session_data.get("base_session_id"),
                            config=config,
                            start_time=datetime.fromisoformat(session_data["start_time"]),
                            last_update_time=datetime.fromisoformat(
                                session_data["last_update_time"]
                            ),
                            total_samples_seen=session_data.get("total_samples_seen", 0),
                            incremental_updates=session_data.get("incremental_updates", 0),
                            performance_history=session_data.get("performance_history", []),
                            data_versions=session_data.get("data_versions", []),
                        )
                        self.active_sessions[session.session_id] = session
            except Exception as e:
                self.logger.warning(f"セッション情報の読み込みに失敗: {e}")

    def _save_sessions(self):
        """セッション情報を保存"""
        try:
            sessions_data = {"sessions": [], "last_updated": datetime.now().isoformat()}

            for session in self.active_sessions.values():
                session_dict = {
                    "session_id": session.session_id,
                    "base_session_id": session.base_session_id,
                    "config": {
                        k: v for k, v in session.config.__dict__.items() if not k.startswith("_")
                    },
                    "start_time": session.start_time.isoformat(),
                    "last_update_time": session.last_update_time.isoformat(),
                    "total_samples_seen": session.total_samples_seen,
                    "incremental_updates": session.incremental_updates,
                    "performance_history": session.performance_history,
                    "data_versions": session.data_versions,
                }
                sessions_data["sessions"].append(session_dict)

            self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.sessions_file, "w") as f:
                json.dump(sessions_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"セッション情報の保存に失敗: {e}")

    def start_continuous_learning(
        self,
        model_type: str,
        config: ContinuousLearningConfig,
        initial_dataset_version: str | None = None,
    ) -> str:
        """
        継続学習を開始

        Args:
            model_type: モデルタイプ（"detection" or "classification"）
            config: 継続学習設定
            initial_dataset_version: 初期データセットバージョン

        Returns:
            セッションID
        """
        import uuid

        session_id = str(uuid.uuid4())
        session = ContinuousLearningSession(
            session_id=session_id,
            base_session_id=None,
            config=config,
            start_time=datetime.now(),
            last_update_time=datetime.now(),
        )

        # 初期学習を実行
        if initial_dataset_version:
            self.logger.info(f"継続学習セッション開始: {session_id}")

            # 初期モデルの学習
            training_config = TrainingConfig(
                model_type=model_type,
                model_name=f"continuous_{model_type}_{session_id}",
                dataset_version_id=initial_dataset_version,
                transfer_learning=bool(config.base_model_path),
                pretrained_model_path=config.base_model_path,
            )

            base_session_id = self.training_manager.start_training(training_config)
            session.base_session_id = base_session_id
            session.data_versions.append(initial_dataset_version)

            # EWC戦略の場合、Fisher情報行列を計算
            if config.strategy == "ewc" and config.base_model_path:
                self._compute_fisher_information(session_id, config.base_model_path)

        self.active_sessions[session_id] = session
        self._save_sessions()

        return session_id

    def add_incremental_data(self, session_id: str, new_data: AnnotationData) -> bool:
        """
        増分データを追加して学習

        Args:
            session_id: セッションID
            new_data: 新しいアノテーションデータ

        Returns:
            学習を実行したかどうか
        """
        session = self.active_sessions.get(session_id)
        if not session:
            self.logger.error(f"セッションが見つかりません: {session_id}")
            return False

        # データ数をカウント
        new_samples = sum(len(video.frames) for video in new_data.video_annotations.values())

        session.total_samples_seen += new_samples

        # 閾値を超えたら学習を実行
        if new_samples >= session.config.incremental_data_threshold:
            self.logger.info(
                f"増分学習を実行: {new_samples}件の新規データ "
                f"(累計: {session.total_samples_seen}件)"
            )

            return self._perform_incremental_update(session_id, new_data)

        self.logger.info(
            f"データを蓄積中: {new_samples}件 (閾値: {session.config.incremental_data_threshold}件)"
        )
        return False

    def _perform_incremental_update(self, session_id: str, new_data: AnnotationData) -> bool:
        """
        増分更新を実行

        Args:
            session_id: セッションID
            new_data: 新しいデータ

        Returns:
            成功したかどうか
        """
        session = self.active_sessions[session_id]

        try:
            # 戦略に応じた学習を実行
            if session.config.strategy == "fine_tuning":
                success = self._fine_tuning_update(session, new_data)
            elif session.config.strategy == "rehearsal":
                success = self._rehearsal_update(session, new_data)
            elif session.config.strategy == "ewc":
                success = self._ewc_update(session, new_data)
            else:
                self.logger.error(f"未対応の学習戦略: {session.config.strategy}")
                return False

            if success:
                session.incremental_updates += 1
                session.last_update_time = datetime.now()
                self._save_sessions()

            return success

        except Exception as e:
            self.logger.error(f"増分更新に失敗: {e}")
            return False

    def _fine_tuning_update(
        self, session: ContinuousLearningSession, new_data: AnnotationData
    ) -> bool:
        """Fine-tuning更新"""
        # 最新のモデルを取得
        latest_model_path = self._get_latest_model_path(session)
        if not latest_model_path:
            self.logger.error("ベースモデルが見つかりません")
            return False

        # 学習設定を作成
        training_config = TrainingConfig(
            model_type=self._get_model_type_from_session(session),
            model_name=f"continuous_update_{session.session_id}_{session.incremental_updates}",
            dataset_version_id=self._create_merged_dataset(session, new_data),
            learning_rate=0.001 * session.config.fine_tuning_lr_factor,
            transfer_learning=True,
            pretrained_model_path=latest_model_path,
        )

        # 学習を実行
        new_session_id = self.training_manager.start_training(training_config)

        # 性能を評価
        performance = self._evaluate_performance(new_session_id)
        session.performance_history.append(
            {
                "update": session.incremental_updates,
                "timestamp": datetime.now().isoformat(),
                "performance": performance,
            }
        )

        return performance >= session.config.min_performance_threshold

    def _rehearsal_update(
        self, session: ContinuousLearningSession, new_data: AnnotationData
    ) -> bool:
        """リハーサル戦略での更新"""
        # リハーサルバッファを更新
        self._update_rehearsal_buffer(session.session_id, new_data)

        # 過去データと新データを混合
        mixed_data = self._create_mixed_dataset(
            session, new_data, self.rehearsal_buffer.get(session.session_id, [])
        )

        # Fine-tuningと同様の処理
        return self._fine_tuning_update(session, mixed_data)

    def _ewc_update(self, session: ContinuousLearningSession, new_data: AnnotationData) -> bool:
        """EWC（Elastic Weight Consolidation）での更新"""
        # TODO: EWC実装
        # Fisher情報行列を使用して重要なパラメータを保護しながら学習
        self.logger.warning("EWC戦略は未実装です")
        return False

    def _compute_fisher_information(self, session_id: str, model_path: str):
        """Fisher情報行列を計算（EWC用）"""
        # TODO: Fisher情報行列の計算実装
        pass

    def _update_rehearsal_buffer(self, session_id: str, new_data: AnnotationData):
        """リハーサルバッファを更新"""
        if session_id not in self.rehearsal_buffer:
            self.rehearsal_buffer[session_id] = []

        # 新しいデータからランダムにサンプリング
        # TODO: 実装
        pass

    def _create_merged_dataset(
        self, session: ContinuousLearningSession, new_data: AnnotationData
    ) -> str:
        """既存データと新データをマージしたデータセットを作成"""
        # TODO: データセットのマージ実装
        return ""

    def _create_mixed_dataset(
        self, session: ContinuousLearningSession, new_data: AnnotationData, rehearsal_data: list
    ) -> AnnotationData:
        """新データと過去データを混合"""
        # TODO: データの混合実装
        return new_data

    def _get_latest_model_path(self, session: ContinuousLearningSession) -> str | None:
        """最新のモデルパスを取得"""
        if session.base_session_id:
            base_session = self.training_manager.active_sessions.get(session.base_session_id)
            if base_session and base_session.best_model_path:
                return base_session.best_model_path
        return session.config.base_model_path

    def _get_model_type_from_session(self, session: ContinuousLearningSession) -> str:
        """セッションからモデルタイプを取得"""
        if session.base_session_id:
            base_session = self.training_manager.active_sessions.get(session.base_session_id)
            if base_session:
                return base_session.config.model_type
        return "classification"  # デフォルト

    def _evaluate_performance(self, session_id: str) -> float:
        """モデルの性能を評価"""
        session_status = self.training_manager.get_session_status(session_id)
        if session_status and session_status.get("final_metrics"):
            metrics = session_status["final_metrics"]
            # 精度またはロスを基準に評価
            if "val_accuracy" in metrics:
                return metrics["val_accuracy"]
            elif "val_loss" in metrics:
                return 1.0 - min(metrics["val_loss"], 1.0)
        return 0.0

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """
        セッション情報を取得

        Args:
            session_id: セッションID

        Returns:
            セッション情報
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "base_session_id": session.base_session_id,
            "config": session.config.__dict__,
            "start_time": session.start_time.isoformat(),
            "last_update_time": session.last_update_time.isoformat(),
            "total_samples_seen": session.total_samples_seen,
            "incremental_updates": session.incremental_updates,
            "performance_history": session.performance_history,
            "data_versions": session.data_versions,
            "status": "active" if session.incremental_updates > 0 else "initialized",
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        """全セッションの一覧を取得"""
        return [self.get_session_info(session_id) for session_id in self.active_sessions]
