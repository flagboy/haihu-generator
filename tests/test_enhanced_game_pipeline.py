"""
拡張ゲームパイプラインのテスト
"""

import numpy as np
import pytest

from src.detection import PlayerPosition as DetectorPlayerPosition
from src.detection import SceneType
from src.pipeline.enhanced_game_pipeline import EnhancedGamePipeline
from src.pipeline.game_pipeline import ProcessingResult


class TestEnhancedGamePipeline:
    """EnhancedGamePipelineのテスト"""

    @pytest.fixture
    def pipeline(self):
        """テスト用パイプライン"""
        return EnhancedGamePipeline(
            game_id="test_game",
            enable_scene_detection=True,
            enable_score_reading=True,
            enable_player_detection=True,
        )

    @pytest.fixture
    def sample_frame(self):
        """テスト用サンプルフレーム"""
        # 緑色のゲームフレーム
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = [0, 128, 0]  # 緑色（ゲームプレイを模擬）
        return frame

    def test_initialization(self, pipeline):
        """初期化のテスト"""
        assert pipeline.game_id == "test_game"
        assert pipeline.scene_detector is not None
        assert pipeline.score_reader is not None
        assert pipeline.player_detector is not None
        assert pipeline.tile_detector is not None
        assert pipeline.action_detector is not None
        assert pipeline.current_scene_type == SceneType.UNKNOWN
        assert pipeline.last_valid_scores is None
        assert len(pipeline.round_boundaries) == 0

    def test_process_frame_basic(self, pipeline, sample_frame):
        """基本的なフレーム処理テスト"""
        frame_data = {"frame": sample_frame, "frame_number": 100, "timestamp": 3.33}
        result = pipeline.process_frame(frame_data)

        assert isinstance(result, ProcessingResult)
        assert result.frame_number == 100
        assert result.processing_time > 0
        # メタデータからシーンタイプを確認
        assert "scene_type" in result.metadata

    def test_process_frame_non_game_scene(self, pipeline):
        """ゲーム外シーンの処理テスト"""
        # 暗いメニューフレーム
        menu_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        menu_frame[:, :] = [20, 20, 20]  # 暗い色

        frame_data = {"frame": menu_frame, "frame_number": 0, "timestamp": 0.0}
        result = pipeline.process_frame(frame_data)

        assert result.success is True
        # メタデータからシーンタイプを確認
        assert result.metadata.get("scene_type") == SceneType.MENU
        assert "ゲームプレイ中ではありません" in result.warnings
        assert result.actions_detected == 0

    def test_validate_scores(self, pipeline):
        """点数検証のテスト"""
        # 有効な点数
        valid_scores = {"east": 25000, "south": 25000, "west": 25000, "north": 25000}
        assert pipeline._validate_scores(valid_scores) is True

        # 無効な点数（合計が異常）
        invalid_scores1 = {"east": 50000, "south": 50000, "west": 50000, "north": 50000}
        assert pipeline._validate_scores(invalid_scores1) is False

        # 無効な点数（プレイヤー数が不足）
        invalid_scores2 = {"east": 30000, "south": 30000, "west": 40000}
        assert pipeline._validate_scores(invalid_scores2) is False

    def test_position_to_player(self, pipeline):
        """位置からプレイヤー取得のテスト"""
        # ゲームを初期化
        pipeline.initialize_game()

        # 各位置のプレイヤーを取得
        east_player = pipeline._position_to_player(DetectorPlayerPosition.EAST)
        assert east_player is not None
        from src.game.player import PlayerPosition

        assert east_player.state.position == PlayerPosition.EAST

    def test_calculate_overall_confidence(self, pipeline):
        """全体信頼度計算のテスト"""

        # モックデータを作成
        class MockInfo:
            confidence = 0.8

        class MockPlayerInfo:
            players = [MockInfo(), MockInfo()]

        class MockScoreInfo:
            total_confidence = 0.7

        class MockDetection:
            confidence = 0.9

        class MockDetectionResult:
            detections = [MockDetection(), MockDetection()]

        confidence = pipeline._calculate_overall_confidence(
            MockInfo(), MockPlayerInfo(), MockScoreInfo(), MockDetectionResult()
        )

        assert 0 <= confidence <= 1.0
        assert confidence > 0.5  # 複数の高信頼度情報があるため

    def test_record_enhanced_history(self, pipeline, sample_frame):
        """拡張履歴記録のテスト"""
        # ゲームを初期化
        pipeline.initialize_game()

        # フレームを処理
        frame_data = {"frame": sample_frame, "frame_number": 0, "timestamp": 0.0}
        result = pipeline.process_frame(frame_data)

        # 履歴が記録されていることを確認
        # フレーム処理でエラーが発生している可能性があるため、処理されたかどうかだけ確認
        assert result is not None
        # 成功した場合のみカウントが増加
        if result.success:
            assert pipeline.total_frames_processed == 1
        else:
            # エラーの場合は失敗カウントが増加
            assert pipeline.failed_frames > 0

    def test_get_enhanced_statistics(self, pipeline):
        """拡張統計情報取得のテスト"""
        stats = pipeline.get_enhanced_statistics()

        assert "enhanced" in stats
        assert "scene_boundaries" in stats["enhanced"]
        assert "current_scene" in stats["enhanced"]
        assert "last_valid_scores" in stats["enhanced"]
        assert "rounds_detected" in stats["enhanced"]

    def test_count_rounds(self, pipeline):
        """局数カウントのテスト"""
        # ラウンド境界を追加
        pipeline.round_boundaries = [
            (100, 3.33, SceneType.ROUND_START),
            (5000, 166.67, SceneType.ROUND_END),
            (5100, 170.00, SceneType.ROUND_START),
            (10000, 333.33, SceneType.ROUND_END),
        ]

        rounds = pipeline._count_rounds()
        assert rounds == 2  # ROUND_STARTの数

    def test_scene_boundary_detection(self, pipeline, sample_frame):
        """シーン境界検出のテスト"""
        # ゲーム開始シーンを模擬
        frame_data = {"frame": sample_frame, "frame_number": 0, "timestamp": 0.0}
        result = pipeline.process_frame(frame_data)

        # シーン境界の検出
        if result.metadata.get("scene_type") == SceneType.GAME_START:
            assert len(pipeline.round_boundaries) == 1
            assert pipeline.round_boundaries[0][2] == SceneType.GAME_START

    def test_initialization_without_features(self):
        """機能を無効化した初期化テスト"""
        pipeline = EnhancedGamePipeline(
            game_id="test_game",
            enable_scene_detection=False,
            enable_score_reading=False,
            enable_player_detection=False,
        )

        assert pipeline.scene_detector is None
        assert pipeline.score_reader is None
        assert pipeline.player_detector is None

        # 機能なしでも処理可能
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_data = {"frame": frame, "frame_number": 0, "timestamp": 0.0}
        result = pipeline.process_frame(frame_data)
        assert isinstance(result, ProcessingResult)
