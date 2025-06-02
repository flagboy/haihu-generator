"""
統計情報の収集を担当するクラス
"""

from datetime import datetime
from typing import Any

import numpy as np

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class StatisticsCollector:
    """統計情報を収集・集計するクラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config_manager = config_manager
        self.config = config_manager._config
        self.logger = get_logger(self.__class__.__name__)

    def collect_statistics(
        self,
        processing_result: Any,
        ai_results: list[dict[str, Any]] | None = None,
        game_results: Any | None = None,
    ) -> dict[str, Any]:
        """
        処理結果から統計情報を収集

        Args:
            processing_result: 処理結果オブジェクト
            ai_results: AI処理結果（オプション）
            game_results: ゲーム処理結果（オプション）

        Returns:
            統計情報の辞書
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_result.processing_time,
            "frame_count": processing_result.frame_count,
            "detected_tiles": processing_result.detected_tiles,
            "success_rate": 1.0 if processing_result.success else 0.0,
        }

        # AI統計を追加
        if ai_results:
            stats["ai_statistics"] = self._collect_ai_statistics(ai_results)

        # ゲーム統計を追加
        if game_results:
            stats["game_statistics"] = self._collect_game_statistics(game_results)

        # パフォーマンス統計を追加
        stats["performance"] = self._collect_performance_statistics(processing_result)

        # エラー情報を追加
        if processing_result.errors:
            stats["errors"] = processing_result.errors

        if processing_result.warnings:
            stats["warnings"] = processing_result.warnings

        return stats

    def _collect_ai_statistics(self, ai_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        AI処理の統計情報を収集

        Args:
            ai_results: AI処理結果のリスト

        Returns:
            AI統計情報
        """
        detection_counts = []
        classification_confidences = []
        tile_types = {}

        for result in ai_results:
            # 検出数を記録
            detections = result.get("detections", [])
            detection_counts.append(len(detections))

            # 分類結果を集計
            for det, cls in result.get("classifications", []):
                classification_confidences.append(cls.confidence)

                # 牌種別の統計
                tile_type = cls.label
                if tile_type not in tile_types:
                    tile_types[tile_type] = {"count": 0, "confidences": [], "positions": []}

                tile_types[tile_type]["count"] += 1
                tile_types[tile_type]["confidences"].append(cls.confidence)
                tile_types[tile_type]["positions"].append(det.bbox)

        # 統計値を計算
        stats = {
            "total_detections": sum(detection_counts),
            "average_detections_per_frame": np.mean(detection_counts) if detection_counts else 0,
            "max_detections_per_frame": max(detection_counts) if detection_counts else 0,
            "min_detections_per_frame": min(detection_counts) if detection_counts else 0,
            "frames_with_detections": sum(1 for c in detection_counts if c > 0),
            "detection_rate": sum(1 for c in detection_counts if c > 0) / len(ai_results)
            if ai_results
            else 0,
        }

        # 信頼度統計
        if classification_confidences:
            stats["confidence_statistics"] = {
                "mean": np.mean(classification_confidences),
                "median": np.median(classification_confidences),
                "std": np.std(classification_confidences),
                "min": min(classification_confidences),
                "max": max(classification_confidences),
                "percentiles": {
                    "25": np.percentile(classification_confidences, 25),
                    "75": np.percentile(classification_confidences, 75),
                    "95": np.percentile(classification_confidences, 95),
                },
            }

        # 牌種別統計
        tile_statistics = {}
        for tile_type, data in tile_types.items():
            tile_statistics[tile_type] = {
                "count": data["count"],
                "average_confidence": np.mean(data["confidences"]) if data["confidences"] else 0,
                "frequency": data["count"] / stats["total_detections"]
                if stats["total_detections"] > 0
                else 0,
            }

        stats["tile_type_distribution"] = tile_statistics

        return stats

    def _collect_game_statistics(self, game_results: Any) -> dict[str, Any]:
        """
        ゲーム処理の統計情報を収集

        Args:
            game_results: ゲーム処理結果

        Returns:
            ゲーム統計情報
        """
        stats = {}

        # GamePipelineResultの場合
        if hasattr(game_results, "get_statistics"):
            return game_results.get_statistics()

        # 基本統計
        if hasattr(game_results, "game_states"):
            stats["total_game_states"] = len(game_results.game_states)
            stats["valid_game_states"] = sum(1 for s in game_results.game_states if s.is_valid)

        # アクション統計
        if hasattr(game_results, "actions"):
            action_counts = {}
            for action in game_results.actions:
                action_type = action.get("type", "unknown")
                action_counts[action_type] = action_counts.get(action_type, 0) + 1

            stats["action_distribution"] = action_counts
            stats["total_actions"] = sum(action_counts.values())

        # ラウンド統計
        if hasattr(game_results, "rounds"):
            stats["total_rounds"] = len(game_results.rounds)
            stats["completed_rounds"] = sum(
                1 for r in game_results.rounds if r.get("completed", False)
            )

        # プレイヤー統計
        if hasattr(game_results, "players"):
            stats["player_count"] = len(game_results.players)

            # 各プレイヤーのスコア変動
            score_changes = []
            for player in game_results.players:
                if "score_history" in player:
                    initial = player["score_history"][0] if player["score_history"] else 25000
                    final = player["score_history"][-1] if player["score_history"] else 25000
                    score_changes.append(final - initial)

            if score_changes:
                stats["score_statistics"] = {
                    "max_gain": max(score_changes),
                    "max_loss": min(score_changes),
                    "average_change": np.mean(score_changes),
                    "std_change": np.std(score_changes),
                }

        return stats

    def _collect_performance_statistics(self, processing_result: Any) -> dict[str, Any]:
        """
        パフォーマンス統計を収集

        Args:
            processing_result: 処理結果

        Returns:
            パフォーマンス統計
        """
        stats = {
            "total_processing_time": processing_result.processing_time,
            "frames_per_second": processing_result.frame_count / processing_result.processing_time
            if processing_result.processing_time > 0
            else 0,
            "tiles_per_second": processing_result.detected_tiles / processing_result.processing_time
            if processing_result.processing_time > 0
            else 0,
        }

        # 処理段階別の時間（利用可能な場合）
        if hasattr(processing_result, "timing_breakdown"):
            stats["timing_breakdown"] = processing_result.timing_breakdown

        # メモリ使用量（利用可能な場合）
        if hasattr(processing_result, "memory_usage"):
            stats["memory_usage"] = processing_result.memory_usage

        # GPU使用率（利用可能な場合）
        if hasattr(processing_result, "gpu_usage"):
            stats["gpu_usage"] = processing_result.gpu_usage

        return stats

    def merge_statistics(self, *stats_dicts: dict[str, Any]) -> dict[str, Any]:
        """
        複数の統計辞書をマージ

        Args:
            *stats_dicts: マージする統計辞書

        Returns:
            マージされた統計辞書
        """
        merged = {}

        for stats in stats_dicts:
            if stats:
                for key, value in stats.items():
                    if key not in merged:
                        merged[key] = value
                    elif isinstance(value, dict) and isinstance(merged[key], dict):
                        # 辞書の場合は再帰的にマージ
                        merged[key] = self.merge_statistics(merged[key], value)
                    elif isinstance(value, list) and isinstance(merged[key], list):
                        # リストの場合は結合
                        merged[key].extend(value)
                    else:
                        # それ以外は後の値で上書き
                        merged[key] = value

        return merged

    def format_statistics_report(self, statistics: dict[str, Any]) -> str:
        """
        統計情報を読みやすい形式にフォーマット

        Args:
            statistics: 統計情報辞書

        Returns:
            フォーマットされたレポート文字列
        """
        lines = ["=" * 60, "処理統計レポート", "=" * 60]

        # 基本情報
        lines.append(f"\n実行時刻: {statistics.get('timestamp', 'N/A')}")
        lines.append(f"処理時間: {statistics.get('processing_time', 0):.2f}秒")
        lines.append(f"フレーム数: {statistics.get('frame_count', 0)}")
        lines.append(f"検出牌数: {statistics.get('detected_tiles', 0)}")

        # AI統計
        if "ai_statistics" in statistics:
            ai_stats = statistics["ai_statistics"]
            lines.append("\n[AI処理統計]")
            lines.append(f"  総検出数: {ai_stats.get('total_detections', 0)}")
            lines.append(
                f"  フレームあたり平均検出数: {ai_stats.get('average_detections_per_frame', 0):.2f}"
            )
            lines.append(f"  検出率: {ai_stats.get('detection_rate', 0):.2%}")

            if "confidence_statistics" in ai_stats:
                conf_stats = ai_stats["confidence_statistics"]
                lines.append(f"  平均信頼度: {conf_stats.get('mean', 0):.3f}")
                lines.append(
                    f"  信頼度範囲: {conf_stats.get('min', 0):.3f} - {conf_stats.get('max', 0):.3f}"
                )

        # ゲーム統計
        if "game_statistics" in statistics:
            game_stats = statistics["game_statistics"]
            lines.append("\n[ゲーム処理統計]")
            lines.append(f"  総アクション数: {game_stats.get('total_actions', 0)}")
            lines.append(f"  ラウンド数: {game_stats.get('total_rounds', 0)}")
            lines.append(f"  プレイヤー数: {game_stats.get('player_count', 0)}")

        # パフォーマンス統計
        if "performance" in statistics:
            perf_stats = statistics["performance"]
            lines.append("\n[パフォーマンス統計]")
            lines.append(f"  処理速度: {perf_stats.get('frames_per_second', 0):.2f} FPS")
            lines.append(f"  牌検出速度: {perf_stats.get('tiles_per_second', 0):.2f} 牌/秒")

        # エラー・警告
        if "errors" in statistics:
            lines.append(f"\n[エラー] {len(statistics['errors'])}件")
            for error in statistics["errors"][:5]:  # 最初の5件のみ表示
                lines.append(f"  - {error}")

        if "warnings" in statistics:
            lines.append(f"\n[警告] {len(statistics['warnings'])}件")
            for warning in statistics["warnings"][:5]:  # 最初の5件のみ表示
                lines.append(f"  - {warning}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
