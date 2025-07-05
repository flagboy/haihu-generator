"""
拡張機能のテスト
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.detection import (
    PlayerDetectionResult,
    PlayerPosition,
    PlayerScore,
    SceneDetectionResult,
    SceneType,
    ScoreReadingResult,
)
from src.detection.cached_scene_detector import CachedSceneDetector
from src.pipeline.batch_enhanced_game_pipeline import BatchEnhancedGamePipeline
from src.pipeline.enhanced_game_pipeline import EnhancedGamePipeline, EnhancedProcessingResult
from src.training.annotation_data import FrameAnnotation
from src.training.enhanced_semi_auto_labeler import (
    EnhancedPredictionResult,
    EnhancedSemiAutoLabeler,
)


class TestEnhancedSemiAutoLabeler:
    """EnhancedSemiAutoLabelerのテスト"""

    @pytest.fixture
    def mock_config_manager(self):
        """ConfigManagerのモック"""
        config_manager = Mock()
        config_manager.get_config.return_value = {
            "scene_detection": {"confidence_threshold": 0.7},
            "score_reading": {"enable_ocr": True},
            "player_detection": {"enable_tracking": True},
            "training": {
                "enhanced_labeling": {
                    "filter_non_game_scenes": True,
                    "min_scene_confidence": 0.7,
                    "use_context_for_classification": True,
                }
            },
            "labeling": {
                "confidence_threshold": 0.5,
                "auto_area_classification": True,
                "enable_occlusion_detection": False,
            },
        }
        return config_manager

    @pytest.fixture
    def sample_frame_annotation(self, tmp_path):
        """サンプルフレームアノテーション"""
        # ダミー画像を作成
        image_path = tmp_path / "test_frame.jpg"
        dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        import cv2

        cv2.imwrite(str(image_path), dummy_image)

        return FrameAnnotation(
            frame_id="frame_0001",
            image_path=str(image_path),
            image_width=1280,
            image_height=720,
            timestamp=1.0,
            tiles=[],
        )

    @patch("src.detection.tile_detector.TileDetector")
    @patch("src.classification.tile_classifier.TileClassifier")
    @patch("src.training.enhanced_semi_auto_labeler.SceneDetector")
    @patch("src.training.enhanced_semi_auto_labeler.ScoreReader")
    @patch("src.training.enhanced_semi_auto_labeler.PlayerDetector")
    def test_enhanced_labeler_initialization(
        self,
        mock_player_detector,
        mock_score_reader,
        mock_scene_detector,
        mock_classifier,
        mock_detector,
        mock_config_manager,
    ):
        """初期化テスト"""
        labeler = EnhancedSemiAutoLabeler(mock_config_manager)

        assert labeler.filter_non_game_scenes is True
        assert labeler.min_scene_confidence == 0.7
        assert labeler.use_context_for_classification is True
        assert mock_scene_detector.called
        assert mock_score_reader.called
        assert mock_player_detector.called

    @patch("src.training.enhanced_semi_auto_labeler.cv2")
    def test_filter_non_game_scenes(self, mock_cv2, mock_config_manager, sample_frame_annotation):
        """非ゲームシーンのフィルタリングテスト"""
        with patch(
            "src.training.enhanced_semi_auto_labeler.SceneDetector"
        ) as mock_scene_detector_class:
            # SceneDetectorのモック設定
            mock_scene_detector = Mock()
            mock_scene_detector_class.return_value = mock_scene_detector

            # メニューシーンを返すように設定
            mock_scene_detector.detect_scene.return_value = SceneDetectionResult(
                scene_type=SceneType.MENU,
                confidence=0.9,
                frame_number=1,
                timestamp=1.0,
                metadata={},
            )

            # 画像読み込みのモック
            mock_cv2.imread.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)

            with (
                patch("src.detection.tile_detector.TileDetector"),
                patch("src.classification.tile_classifier.TileClassifier"),
            ):
                labeler = EnhancedSemiAutoLabeler(mock_config_manager)

                # 予測を実行
                result = labeler.predict_frame_annotations(sample_frame_annotation)

                # 結果を確認
                assert isinstance(result, EnhancedPredictionResult)
                assert result.frame_annotation.is_valid is False
                assert len(result.frame_annotation.tiles) == 0
                assert result.scene_result.scene_type == SceneType.MENU


class TestEnhancedGamePipeline:
    """EnhancedGamePipelineのテスト"""

    def test_enhanced_pipeline_initialization(self):
        """初期化テスト"""
        pipeline = EnhancedGamePipeline(
            game_id="test_game",
            enable_scene_detection=True,
            enable_score_reading=True,
            enable_player_detection=True,
        )

        assert pipeline.game_id == "test_game"
        assert pipeline.scene_detector is not None
        assert pipeline.score_reader is not None
        assert pipeline.player_detector is not None

    def test_process_frame_enhanced(self):
        """フレーム処理テスト"""
        pipeline = EnhancedGamePipeline()

        # テスト用フレーム
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 各検出器をモック
        with (
            patch.object(pipeline.scene_detector, "detect_scene") as mock_scene,
            patch.object(pipeline.player_detector, "detect_players") as mock_player,
            patch.object(pipeline.score_reader, "read_scores") as mock_score,
            patch.object(pipeline.tile_detector, "detect_tiles") as mock_tiles,
        ):
            # モックの戻り値を設定
            mock_scene.return_value = SceneDetectionResult(
                scene_type=SceneType.GAME_PLAY,
                confidence=0.9,
                frame_number=1,
                timestamp=1.0,
                metadata={},
            )

            mock_player.return_value = PlayerDetectionResult(
                players=[],
                active_position=PlayerPosition.SOUTH,
                dealer_position=PlayerPosition.EAST,
                round_wind="east",
            )

            mock_score.return_value = ScoreReadingResult(
                scores=[
                    PlayerScore(
                        player_position="east",
                        score=25000,
                        confidence=0.9,
                        bbox=(0, 0, 100, 50),
                    ),
                    PlayerScore(
                        player_position="south",
                        score=25000,
                        confidence=0.9,
                        bbox=(0, 0, 100, 50),
                    ),
                    PlayerScore(
                        player_position="west",
                        score=25000,
                        confidence=0.9,
                        bbox=(0, 0, 100, 50),
                    ),
                    PlayerScore(
                        player_position="north",
                        score=25000,
                        confidence=0.9,
                        bbox=(0, 0, 100, 50),
                    ),
                ],
                total_confidence=0.9,
                frame_number=1,
                timestamp=1.0,
            )

            mock_tiles.return_value = Mock(detections=[])

            # 処理を実行
            result = pipeline.process_frame_enhanced(frame, 1, 1.0)

            # 結果を確認
            assert isinstance(result, EnhancedProcessingResult)
            assert result.success is True
            assert result.scene_type == SceneType.GAME_PLAY
            assert result.player_scores == {
                "east": 25000,
                "south": 25000,
                "west": 25000,
                "north": 25000,
            }


class TestBatchEnhancedGamePipeline:
    """BatchEnhancedGamePipelineのテスト"""

    def test_batch_pipeline_initialization(self):
        """初期化テスト"""
        config = {
            "batch_processing": {
                "batch_size": 5,
                "max_workers": 2,
                "enable_parallel": True,
            }
        }

        pipeline = BatchEnhancedGamePipeline(config)

        assert pipeline.batch_size == 5
        assert pipeline.max_workers == 2
        assert pipeline.enable_parallel is True

    def test_process_frames_batch(self):
        """バッチ処理テスト"""
        config = {
            "batch_processing": {
                "batch_size": 2,
                "max_workers": 1,
                "enable_parallel": False,
            }
        }

        pipeline = BatchEnhancedGamePipeline(config)

        # テスト用フレーム
        frames = [(np.zeros((720, 1280, 3), dtype=np.uint8), i, i * 0.033) for i in range(3)]

        with patch.object(pipeline, "process_frame_batch") as mock_process:
            mock_process.return_value = EnhancedProcessingResult(
                success=True,
                frame_number=0,
                actions_detected=0,
                confidence=0.8,
                processing_time=0.1,
            )

            results = pipeline.process_frames_batch(frames)

            assert len(results) == 3
            assert all(isinstance(r, EnhancedProcessingResult) for r in results)
            assert mock_process.call_count == 3


class TestCachedSceneDetector:
    """CachedSceneDetectorのテスト"""

    def test_cached_detector_initialization(self):
        """初期化テスト"""
        config = {
            "cache": {
                "enabled": True,
                "size": 50,
            }
        }

        detector = CachedSceneDetector(config)

        assert detector.enable_cache is True
        assert detector.cache_size == 50
        assert len(detector._frame_hash_cache) == 0

    def test_cache_functionality(self):
        """キャッシュ機能テスト"""
        config = {"cache": {"enabled": True, "size": 10}}
        detector = CachedSceneDetector(config)

        # 同じフレームを2回処理
        frame = np.ones((100, 100, 3), dtype=np.uint8)

        with patch.object(detector, "_detect_scene_core"):
            # 基底クラスのdetect_sceneメソッドをモック
            mock_result = SceneDetectionResult(
                scene_type=SceneType.GAME_PLAY,
                confidence=0.9,
                frame_number=1,
                timestamp=1.0,
                metadata={},
            )

            # 親クラスのメソッドをモック
            with patch(
                "src.detection.cached_scene_detector.SceneDetector.detect_scene",
                return_value=mock_result,
            ):
                # 1回目の呼び出し
                result1 = detector.detect_scene(frame, 1, 1.0)

                # 2回目の呼び出し（キャッシュから取得）
                result2 = detector.detect_scene(frame, 2, 2.0)

                # 結果を確認
                assert result1.scene_type == SceneType.GAME_PLAY
                assert result2.scene_type == SceneType.GAME_PLAY
                assert result2.frame_number == 2  # フレーム番号は更新される
                assert result2.timestamp == 2.0  # タイムスタンプも更新される

    def test_cache_clear(self):
        """キャッシュクリアテスト"""
        detector = CachedSceneDetector()

        # キャッシュにデータを追加
        detector._frame_hash_cache["test_hash"] = {
            "scene_type": SceneType.GAME_PLAY,
            "confidence": 0.9,
            "metadata": {},
        }

        assert len(detector._frame_hash_cache) == 1

        # キャッシュをクリア
        detector.clear_cache()

        assert len(detector._frame_hash_cache) == 0
