"""
StatisticsCollectorのテスト
"""

from unittest.mock import Mock

import pytest

from src.integration.statistics_collector import StatisticsCollector
from src.utils.config import ConfigManager


class TestStatisticsCollector:
    """StatisticsCollectorのテストクラス"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのモック"""
        config_manager = Mock(spec=ConfigManager)
        config_manager._config = {"system": {}, "performance": {}}
        return config_manager

    @pytest.fixture
    def statistics_collector(self, config_manager):
        """StatisticsCollectorのフィクスチャ"""
        return StatisticsCollector(config_manager)

    @pytest.fixture
    def sample_processing_result(self):
        """サンプル処理結果"""
        result = Mock()
        result.success = True
        result.processing_time = 45.6
        result.frame_count = 100
        result.detected_tiles = 250
        result.errors = []
        result.warnings = ["Warning 1", "Warning 2"]
        return result

    @pytest.fixture
    def sample_ai_results(self):
        """サンプルAI結果"""
        return [
            {
                "frame_id": i,
                "detections": [Mock(bbox=[10, 10, 50, 50]) for _ in range(3)],
                "classifications": [
                    (
                        Mock(bbox=[10, 10, 50, 50]),
                        Mock(label=f"{i % 9 + 1}m", confidence=0.8 + i * 0.01),
                    )
                    for _ in range(3)
                ],
            }
            for i in range(10)
        ]

    def test_initialization(self, statistics_collector):
        """初期化テスト"""
        assert statistics_collector is not None
        assert statistics_collector.config is not None

    def test_collect_statistics_basic(self, statistics_collector, sample_processing_result):
        """基本的な統計収集テスト"""
        stats = statistics_collector.collect_statistics(sample_processing_result)

        # 基本情報の確認
        assert "timestamp" in stats
        assert stats["processing_time"] == 45.6
        assert stats["frame_count"] == 100
        assert stats["detected_tiles"] == 250
        assert stats["success_rate"] == 1.0
        assert stats["warnings"] == ["Warning 1", "Warning 2"]

    def test_collect_statistics_with_ai_results(
        self, statistics_collector, sample_processing_result, sample_ai_results
    ):
        """AI結果を含む統計収集テスト"""
        stats = statistics_collector.collect_statistics(
            sample_processing_result, ai_results=sample_ai_results
        )

        # AI統計の確認
        assert "ai_statistics" in stats
        ai_stats = stats["ai_statistics"]
        assert ai_stats["total_detections"] == 30  # 10フレーム × 3検出
        assert ai_stats["frames_with_detections"] == 10
        assert "average_detections_per_frame" in ai_stats
        assert "confidence_statistics" in ai_stats

        # 信頼度統計の確認
        conf_stats = ai_stats["confidence_statistics"]
        assert "mean" in conf_stats
        assert "median" in conf_stats
        assert "std" in conf_stats
        assert "min" in conf_stats
        assert "max" in conf_stats
        assert "percentiles" in conf_stats

    def test_collect_ai_statistics_detailed(self, statistics_collector, sample_ai_results):
        """AI統計収集の詳細テスト"""
        ai_stats = statistics_collector._collect_ai_statistics(sample_ai_results)

        # 統計値の検証
        assert ai_stats["total_detections"] == 30
        assert ai_stats["average_detections_per_frame"] == 3.0
        assert ai_stats["max_detections_per_frame"] == 3
        assert ai_stats["min_detections_per_frame"] == 3
        assert ai_stats["detection_rate"] == 1.0

        # 牌種別分布の確認
        assert "tile_type_distribution" in ai_stats
        tile_dist = ai_stats["tile_type_distribution"]
        assert len(tile_dist) > 0

        # 各牌種の統計情報を確認
        for _tile_type, tile_stats in tile_dist.items():
            assert "count" in tile_stats
            assert "average_confidence" in tile_stats
            assert "frequency" in tile_stats
            assert tile_stats["count"] > 0
            assert 0 <= tile_stats["average_confidence"] <= 1
            assert 0 <= tile_stats["frequency"] <= 1

    def test_collect_game_statistics(self, statistics_collector):
        """ゲーム統計収集のテスト"""
        # ゲーム結果のモック
        game_results = Mock()
        game_results.get_statistics.return_value = {"rounds": 8, "actions": 200, "completed": True}

        game_stats = statistics_collector._collect_game_statistics(game_results)

        assert game_stats["rounds"] == 8
        assert game_stats["actions"] == 200
        assert game_stats["completed"] is True

    def test_collect_game_statistics_without_method(self, statistics_collector):
        """get_statisticsメソッドを持たないゲーム結果の処理テスト"""
        # 属性ベースのゲーム結果
        game_results = Mock(spec=["game_states", "actions", "rounds", "players"])
        game_results.game_states = [Mock(is_valid=True) for _ in range(5)]
        game_results.actions = [
            {"type": "draw"},
            {"type": "draw"},
            {"type": "discard"},
            {"type": "discard"},
            {"type": "chi"},
        ]
        game_results.rounds = [{"completed": True}, {"completed": False}]
        game_results.players = [
            {"score_history": [25000, 26000, 28000]},
            {"score_history": [25000, 24000, 22000]},
        ]

        game_stats = statistics_collector._collect_game_statistics(game_results)

        assert game_stats["total_game_states"] == 5
        assert game_stats["valid_game_states"] == 5
        assert game_stats["total_actions"] == 5
        assert game_stats["action_distribution"]["draw"] == 2
        assert game_stats["action_distribution"]["discard"] == 2
        assert game_stats["action_distribution"]["chi"] == 1
        assert game_stats["total_rounds"] == 2
        assert game_stats["completed_rounds"] == 1
        assert game_stats["player_count"] == 2
        assert "score_statistics" in game_stats

    def test_collect_performance_statistics(self, statistics_collector, sample_processing_result):
        """パフォーマンス統計収集のテスト"""
        # 追加の属性を設定
        sample_processing_result.timing_breakdown = {
            "video_extraction": 10.5,
            "ai_processing": 30.2,
            "game_tracking": 4.9,
        }
        sample_processing_result.memory_usage = {"peak_mb": 512.3, "average_mb": 256.7}
        sample_processing_result.gpu_usage = {"utilization": 85.5, "memory_used_mb": 2048}

        perf_stats = statistics_collector._collect_performance_statistics(sample_processing_result)

        assert perf_stats["total_processing_time"] == 45.6
        assert perf_stats["frames_per_second"] > 0
        assert perf_stats["tiles_per_second"] > 0
        assert perf_stats["timing_breakdown"] == sample_processing_result.timing_breakdown
        assert perf_stats["memory_usage"] == sample_processing_result.memory_usage
        assert perf_stats["gpu_usage"] == sample_processing_result.gpu_usage

    def test_merge_statistics(self, statistics_collector):
        """統計マージのテスト"""
        stats1 = {"a": 1, "b": {"x": 10, "y": 20}, "c": [1, 2, 3]}

        stats2 = {"b": {"y": 30, "z": 40}, "c": [4, 5], "d": "new"}

        stats3 = {"a": 2, "e": {"nested": True}}

        merged = statistics_collector.merge_statistics(stats1, stats2, stats3)

        # マージ結果の確認
        assert merged["a"] == 2  # 最後の値で上書き
        assert merged["b"]["x"] == 10  # 元の値を保持
        assert merged["b"]["y"] == 30  # 上書き
        assert merged["b"]["z"] == 40  # 新規追加
        assert merged["c"] == [1, 2, 3, 4, 5]  # リストは結合
        assert merged["d"] == "new"
        assert merged["e"]["nested"] is True

    def test_format_statistics_report(self, statistics_collector):
        """統計レポートフォーマットのテスト"""
        statistics = {
            "timestamp": "2024-01-01T12:00:00",
            "processing_time": 123.45,
            "frame_count": 500,
            "detected_tiles": 1250,
            "ai_statistics": {
                "total_detections": 1250,
                "average_detections_per_frame": 2.5,
                "detection_rate": 0.95,
                "confidence_statistics": {"mean": 0.85, "min": 0.65, "max": 0.98},
            },
            "game_statistics": {"total_actions": 450, "total_rounds": 8, "player_count": 4},
            "performance": {"frames_per_second": 4.05, "tiles_per_second": 10.12},
            "errors": ["Error 1", "Error 2"],
            "warnings": ["Warning 1"],
        }

        report = statistics_collector.format_statistics_report(statistics)

        # レポートに必要な情報が含まれているか確認
        assert "処理統計レポート" in report
        assert "実行時刻: 2024-01-01T12:00:00" in report
        assert "処理時間: 123.45秒" in report
        assert "フレーム数: 500" in report
        assert "検出牌数: 1250" in report
        assert "[AI処理統計]" in report
        assert "総検出数: 1250" in report
        assert "検出率: 95.00%" in report
        assert "[ゲーム処理統計]" in report
        assert "総アクション数: 450" in report
        assert "[パフォーマンス統計]" in report
        assert "処理速度: 4.05 FPS" in report
        assert "[エラー] 2件" in report
        assert "[警告] 1件" in report

    def test_empty_ai_results(self, statistics_collector):
        """空のAI結果での統計収集テスト"""
        ai_stats = statistics_collector._collect_ai_statistics([])

        assert ai_stats["total_detections"] == 0
        assert ai_stats["average_detections_per_frame"] == 0
        assert ai_stats["detection_rate"] == 0
        assert "confidence_statistics" not in ai_stats
        assert ai_stats["tile_type_distribution"] == {}

    def test_error_handling(self, statistics_collector):
        """エラーハンドリングのテスト"""
        # 不正な処理結果
        invalid_result = Mock()
        invalid_result.processing_time = None  # Noneを設定
        invalid_result.frame_count = "invalid"  # 不正な型
        invalid_result.success = True
        invalid_result.detected_tiles = 100
        invalid_result.errors = None
        invalid_result.warnings = None

        # エラーが発生しないことを確認
        stats = statistics_collector.collect_statistics(invalid_result)
        assert "timestamp" in stats
        assert isinstance(stats, dict)
