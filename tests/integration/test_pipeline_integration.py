"""
パイプライン間の統合テスト

AIPipeline、GamePipeline、VideoProcessorの連携をテスト
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.game.state import FrameState
from src.orchestrator import VideoProcessingOrchestrator
from src.output.tenhou_game_data import (
    ActionType,
    TenhouAction,
    TenhouTile,
)
from src.output.tenhou_json_formatter import TenhouJsonFormatter
from src.pipeline.ai_pipeline import AIPipeline
from src.pipeline.game_pipeline import GamePipeline
from src.utils.config import ConfigManager
from src.video.video_processor import VideoProcessor


class TestPipelineIntegration:
    """パイプライン統合テスト"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクト"""
        return ConfigManager()

    @pytest.fixture
    def sample_frame(self):
        """サンプルフレーム画像"""
        # 1280x720のダミー画像を作成
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # テスト用のマーカーを追加
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        return frame

    @pytest.fixture
    def mock_detection_results(self):
        """モック検出結果"""
        return [
            {"bbox": [100, 100, 50, 70], "confidence": 0.95, "class": 0},
            {"bbox": [200, 100, 50, 70], "confidence": 0.92, "class": 1},
            {"bbox": [300, 100, 50, 70], "confidence": 0.88, "class": 2},
        ]

    def test_ai_to_game_pipeline_flow(self, config_manager, sample_frame, mock_detection_results):
        """AIPipelineからGamePipelineへのデータフロー"""
        # AIPipelineのモック設定
        with patch("src.ai_pipeline.TileDetector") as MockDetector:
            mock_detector = Mock()
            mock_detector.detect_tiles.return_value = mock_detection_results
            MockDetector.return_value = mock_detector

            with patch("src.ai_pipeline.TileClassifier") as MockClassifier:
                mock_classifier = Mock()
                # 分類結果をモック
                mock_classifier.classify_tiles_batch.return_value = [
                    {"class": "1m", "confidence": 0.95},
                    {"class": "2m", "confidence": 0.92},
                    {"class": "3m", "confidence": 0.88},
                ]
                MockClassifier.return_value = mock_classifier

                # AIPipelineを実行
                ai_pipeline = AIPipeline(config_manager)
                ai_result = ai_pipeline.process_frame(sample_frame, frame_number=1)

                assert ai_result is not None
                assert len(ai_result.tiles) == 3

                # GamePipelineでAIの結果を処理
                game_pipeline = GamePipeline(config_manager)
                game_pipeline.initialize_game(
                    ["プレイヤー1", "プレイヤー2", "プレイヤー3", "プレイヤー4"]
                )

                # フレーム状態を作成
                FrameState(
                    frame_number=1,
                    timestamp=0.033,
                    detected_tiles={
                        "player_0_hand": [
                            TenhouTile(tile_type="1m", tile_id=0),
                            TenhouTile(tile_type="2m", tile_id=1),
                            TenhouTile(tile_type="3m", tile_id=2),
                        ]
                    },
                )

                # ゲーム状態を更新
                success = game_pipeline.process_frame(ai_result, frame_number=1, timestamp=0.033)
                assert success

    def test_video_to_ai_pipeline_flow(self, config_manager, sample_frame):
        """VideoProcessorからAIPipelineへのデータフロー"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            # ビデオファイルをモック
            with patch("cv2.VideoCapture") as MockCapture:
                mock_cap = Mock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.side_effect = lambda x: {
                    cv2.CAP_PROP_FRAME_COUNT: 10,
                    cv2.CAP_PROP_FPS: 30,
                    cv2.CAP_PROP_FRAME_WIDTH: 1280,
                    cv2.CAP_PROP_FRAME_HEIGHT: 720,
                }.get(x, 0)
                mock_cap.read.side_effect = [(True, sample_frame)] * 5 + [(False, None)]
                MockCapture.return_value = mock_cap

                # VideoProcessorでフレームを抽出
                video_processor = VideoProcessor(config_manager)
                frames = video_processor.extract_frames(video_path, frame_interval=1)

                assert len(frames) == 5

                # AIPipelineで処理できることを確認
                with patch("src.ai_pipeline.TileDetector"), patch("src.ai_pipeline.TileClassifier"):
                    ai_pipeline = AIPipeline(config_manager)
                    for frame_data in frames:
                        result = ai_pipeline.process_frame(
                            frame_data["frame"], frame_number=frame_data["frame_number"]
                        )
                        assert result is not None

        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_complete_pipeline_integration(self, config_manager):
        """完全なパイプライン統合テスト"""
        # VideoProcessor → AIPipeline → GamePipeline → TenhouJson
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            # 全パイプラインをモック
            with patch("cv2.VideoCapture") as MockCapture:
                # VideoProcessorのモック
                mock_cap = Mock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.side_effect = lambda x: {
                    cv2.CAP_PROP_FRAME_COUNT: 10,
                    cv2.CAP_PROP_FPS: 30,
                }.get(x, 0)
                # 3フレーム分のデータ
                mock_cap.read.side_effect = [
                    (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
                    (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
                    (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
                    (False, None),
                ]
                MockCapture.return_value = mock_cap

                # AIPipelineのモック
                with patch("src.ai_pipeline.TileDetector") as MockDetector:
                    mock_detector = Mock()
                    mock_detector.detect_tiles.return_value = [
                        {"bbox": [100, 100, 50, 70], "confidence": 0.95, "class": 0}
                    ]
                    MockDetector.return_value = mock_detector

                    with patch("src.ai_pipeline.TileClassifier") as MockClassifier:
                        mock_classifier = Mock()
                        mock_classifier.classify_tiles_batch.return_value = [
                            {"class": "1m", "confidence": 0.95}
                        ]
                        MockClassifier.return_value = mock_classifier

                        # Orchestratorを使用して全体を実行
                        orchestrator = VideoProcessingOrchestrator(config_manager)
                        result = orchestrator.process_video(video_path)

                        assert result.success
                        assert result.frames_processed == 3

        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_game_state_to_tenhou_format(self, config_manager):
        """GameStateからTenhouフォーマットへの変換テスト"""
        # GamePipelineでゲームデータを作成
        game_pipeline = GamePipeline(config_manager)
        game_pipeline.initialize_game(["東家", "南家", "西家", "北家"])

        # アクションを追加
        game_pipeline.game_data.add_action(
            TenhouAction(
                action_type=ActionType.DRAW,
                player=0,
                tile=TenhouTile(tile_type="1m", tile_id=0),
                timestamp=1.0,
            )
        )
        game_pipeline.game_data.add_action(
            TenhouAction(
                action_type=ActionType.DISCARD,
                player=0,
                tile=TenhouTile(tile_type="9p", tile_id=35),
                timestamp=2.0,
                is_tsumogiri=True,
            )
        )

        # Tenhou形式に変換
        formatter = TenhouJsonFormatter()
        tenhou_json = formatter.format_game_data(game_pipeline.game_data, validate=True)

        assert tenhou_json is not None
        assert "title" in tenhou_json
        assert "name" in tenhou_json
        assert len(tenhou_json["name"]) == 4
        assert "log" in tenhou_json
        assert len(tenhou_json["log"]) == 2

    def test_error_handling_across_pipelines(self, config_manager):
        """パイプライン間のエラーハンドリングテスト"""
        # 不正な入力でのエラーハンドリング
        ai_pipeline = AIPipeline(config_manager)
        game_pipeline = GamePipeline(config_manager)

        # 空のフレームでAIPipelineを実行
        result = ai_pipeline.process_frame(None, frame_number=1)
        assert result.tiles == []  # エラーでも空の結果を返す

        # 初期化前のGamePipeline実行
        success = game_pipeline.process_frame(result, frame_number=1, timestamp=0.0)
        assert not success  # 初期化前なので失敗

        # 初期化後は成功
        game_pipeline.initialize_game(["P1", "P2", "P3", "P4"])
        success = game_pipeline.process_frame(result, frame_number=1, timestamp=0.0)
        assert success

    def test_pipeline_performance_metrics(self, config_manager):
        """パイプラインのパフォーマンスメトリクステスト"""
        import time

        # 各パイプラインの処理時間を計測
        timings = {}

        # VideoProcessor
        start = time.time()
        VideoProcessor(config_manager)
        timings["video_init"] = time.time() - start

        # AIPipeline
        start = time.time()
        with patch("src.ai_pipeline.TileDetector"), patch("src.ai_pipeline.TileClassifier"):
            AIPipeline(config_manager)
        timings["ai_init"] = time.time() - start

        # GamePipeline
        start = time.time()
        GamePipeline(config_manager)
        timings["game_init"] = time.time() - start

        # 初期化時間が妥当であることを確認
        for name, timing in timings.items():
            assert timing < 1.0, f"{name} took too long: {timing}s"

    @pytest.mark.parametrize(
        "frame_count,expected_actions",
        [
            (10, 5),  # 10フレームで5アクション
            (30, 15),  # 30フレームで15アクション
            (100, 50),  # 100フレームで50アクション
        ],
    )
    def test_scalability(self, config_manager, frame_count, expected_actions):
        """スケーラビリティテスト"""
        # 大量のフレームでの動作確認
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(frame_count)]

        with patch("src.ai_pipeline.TileDetector"), patch("src.ai_pipeline.TileClassifier"):
            ai_pipeline = AIPipeline(config_manager)
            game_pipeline = GamePipeline(config_manager)
            game_pipeline.initialize_game(["P1", "P2", "P3", "P4"])

            processed_count = 0
            for i, frame in enumerate(frames):
                result = ai_pipeline.process_frame(frame, frame_number=i)
                if game_pipeline.process_frame(result, frame_number=i, timestamp=i * 0.033):
                    processed_count += 1

            assert processed_count == frame_count

            # 統計情報の確認
            stats = game_pipeline.get_statistics()
            assert stats["frames_processed"] == frame_count
