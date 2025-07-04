"""
継続学習APIルート

継続学習システムのREST APIエンドポイント
"""

from flask import Blueprint, jsonify, request

from src.training.annotation_data import AnnotationData
from src.training.dataset_manager import DatasetManager
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

from ..components.checkpoint_manager import CheckpointManager
from ..components.data_history_manager import DataHistoryManager
from ..continuous_learning_controller import (
    ContinuousLearningConfig,
    ContinuousLearningController,
)
from ..training_manager import TrainingManager

# ブループリント作成
continuous_learning_bp = Blueprint("continuous_learning", __name__)

# ロガー
logger = get_logger(__name__)

# グローバルインスタンス（実際のアプリケーションでは適切に管理）
config_manager = ConfigManager()
training_manager = TrainingManager(config_manager)
dataset_manager = DatasetManager(config_manager)
checkpoint_manager = CheckpointManager()
data_history_manager = DataHistoryManager()
cl_controller = ContinuousLearningController(training_manager, dataset_manager, checkpoint_manager)


@continuous_learning_bp.route("/sessions", methods=["POST"])
def start_continuous_learning_session():
    """継続学習セッションを開始"""
    try:
        data = request.get_json()

        # 設定を作成
        config = ContinuousLearningConfig(
            base_model_path=data.get("base_model_path"),
            incremental_data_threshold=data.get("incremental_data_threshold", 100),
            strategy=data.get("strategy", "fine_tuning"),
            fine_tuning_lr_factor=data.get("fine_tuning_lr_factor", 0.1),
            freeze_layers=data.get("freeze_layers", []),
            rehearsal_size=data.get("rehearsal_size", 1000),
            rehearsal_ratio=data.get("rehearsal_ratio", 0.3),
            ewc_lambda=data.get("ewc_lambda", 0.5),
            fisher_samples=data.get("fisher_samples", 200),
            use_knowledge_distillation=data.get("use_knowledge_distillation", False),
            distillation_temperature=data.get("distillation_temperature", 3.0),
            distillation_alpha=data.get("distillation_alpha", 0.7),
            auto_train_enabled=data.get("auto_train_enabled", True),
            check_interval_hours=data.get("check_interval_hours", 24),
            min_performance_threshold=data.get("min_performance_threshold", 0.85),
            data_versioning=data.get("data_versioning", True),
            max_data_versions=data.get("max_data_versions", 5),
        )

        # セッションを開始
        session_id = cl_controller.start_continuous_learning(
            model_type=data["model_type"],
            config=config,
            initial_dataset_version=data.get("initial_dataset_version"),
        )

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "message": "継続学習セッションを開始しました",
            }
        ), 201

    except Exception as e:
        logger.error(f"継続学習セッション開始エラー: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@continuous_learning_bp.route("/sessions/<session_id>/data", methods=["POST"])
def add_incremental_data(session_id: str):
    """増分データを追加"""
    try:
        data = request.get_json()

        # アノテーションデータを作成（実際の実装では適切にパース）
        annotation_data = AnnotationData()
        # ここでデータをパース

        # データを追加
        trained = cl_controller.add_incremental_data(session_id, annotation_data)

        # データ履歴を記録
        if "sample_ids" in data:
            data_history_manager.add_samples(data["samples"])
            data_history_manager.record_usage(
                session_id=session_id,
                dataset_version=data.get("dataset_version", "unknown"),
                sample_ids=data["sample_ids"],
                performance_metrics={},
                strategy=cl_controller.active_sessions[session_id].config.strategy,
            )

        return jsonify(
            {
                "success": True,
                "trained": trained,
                "message": "データを追加しました" + ("（学習を実行）" if trained else ""),
            }
        )

    except Exception as e:
        logger.error(f"データ追加エラー: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@continuous_learning_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session_info(session_id: str):
    """セッション情報を取得"""
    try:
        info = cl_controller.get_session_info(session_id)

        if not info:
            return jsonify({"success": False, "error": "セッションが見つかりません"}), 404

        # データ履歴情報を追加
        performance_trend = data_history_manager.get_performance_trend(
            session_id=session_id, last_n_entries=10
        )
        info["performance_trend"] = performance_trend

        return jsonify({"success": True, "session": info})

    except Exception as e:
        logger.error(f"セッション情報取得エラー: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@continuous_learning_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """セッション一覧を取得"""
    try:
        sessions = cl_controller.list_sessions()

        return jsonify({"success": True, "sessions": sessions, "total": len(sessions)})

    except Exception as e:
        logger.error(f"セッション一覧取得エラー: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@continuous_learning_bp.route("/rehearsal-samples", methods=["POST"])
def select_rehearsal_samples():
    """リハーサル用サンプルを選択"""
    try:
        data = request.get_json()

        selected_samples = data_history_manager.select_rehearsal_samples(
            available_samples=data["available_samples"],
            num_samples=data["num_samples"],
            selection_strategy=data.get("selection_strategy", "importance_sampling"),
        )

        # 重要度スコアも返す
        importance_scores = data_history_manager.get_sample_importance_scores(
            selected_samples, method="performance"
        )

        return jsonify(
            {
                "success": True,
                "selected_samples": selected_samples,
                "importance_scores": importance_scores,
            }
        )

    except Exception as e:
        logger.error(f"リハーサルサンプル選択エラー: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@continuous_learning_bp.route("/performance-trend", methods=["GET"])
def get_performance_trend():
    """性能トレンドを取得"""
    try:
        session_id = request.args.get("session_id")
        last_n = int(request.args.get("last_n", 20))

        trend = data_history_manager.get_performance_trend(
            session_id=session_id, last_n_entries=last_n
        )

        return jsonify({"success": True, "trend": trend})

    except Exception as e:
        logger.error(f"性能トレンド取得エラー: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@continuous_learning_bp.route("/data-cleanup", methods=["POST"])
def cleanup_old_data():
    """古いデータをクリーンアップ"""
    try:
        days_to_keep = request.get_json().get("days_to_keep", 90)

        data_history_manager.cleanup_old_data(days_to_keep)

        return jsonify(
            {"success": True, "message": f"{days_to_keep}日より古いデータをクリーンアップしました"}
        )

    except Exception as e:
        logger.error(f"データクリーンアップエラー: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
