"""
機械学習改善機能のテスト
"""

import numpy as np

from src.detection import SceneDetector, SceneType
from src.detection.cached_scene_detector import CachedSceneDetector
from src.detection.tile_detector import TileDetector
from src.detection.types import SceneMetadata
from src.pipeline.batch_enhanced_game_pipeline import BatchEnhancedGamePipeline
from src.training.enhanced_semi_auto_labeler import EnhancedSemiAutoLabeler
from src.utils.config import ConfigManager


class TestTileDetectorIntegration:
    """TileDetector統合のテスト"""

    def setup_method(self):
        """テストのセットアップ"""
        self.config = ConfigManager()
        self.scene_detector = SceneDetector()
        self.tile_detector = TileDetector(self.config)

    def test_tile_arrangement_detection(self):
        """牌配置によるシーン検出のテスト"""
        # ダミーフレームを作成
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # _detect_by_tile_arrangementメソッドのテスト
        scene_type, confidence, metadata = self.scene_detector._detect_by_tile_arrangement(frame)

        assert isinstance(scene_type, SceneType)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(metadata, dict)
        assert "total_tiles" in metadata

    def test_tile_classification_with_context(self):
        """コンテキスト情報を使った牌分類のテスト"""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = []

        # シーンコンテキストを作成
        scene_context = {
            "scene_type": "ROUND_START",
            "player_positions": {
                "south": {"bbox": (0, 0, 100, 100), "is_dealer": True, "is_active": True}
            },
        }

        # コンテキスト付き分類のテスト
        areas = self.tile_detector.classify_tile_areas(frame, detections, scene_context)

        assert isinstance(areas, dict)
        assert "hand_tiles" in areas
        assert "discarded_tiles" in areas
        assert "called_tiles" in areas


class TestEnhancedSemiAutoLabeler:
    """拡張半自動ラベラーのテスト"""

    def setup_method(self):
        """テストのセットアップ"""
        self.config = ConfigManager()
        self.labeler = EnhancedSemiAutoLabeler(self.config)

    def test_scene_filtering(self, tmp_path):
        """シーンフィルタリングのテスト"""
        from src.training.annotation_data import FrameAnnotation

        # テスト用フレームアノテーションを作成
        frame_annotations = []
        for i in range(5):
            # ダミー画像を作成
            import cv2

            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            image_path = tmp_path / f"frame_{i}.jpg"
            cv2.imwrite(str(image_path), dummy_image)

            frame = FrameAnnotation(
                frame_id=f"frame_{i}",
                image_path=str(image_path),
                image_width=100,
                image_height=100,
                timestamp=float(i),
                tiles=[],
            )
            frame_annotations.append(frame)

        # フィルタリングのテスト
        filtered = self.labeler.filter_game_frames(frame_annotations)

        assert isinstance(filtered, list)
        assert len(filtered) <= len(frame_annotations)

    def test_enhanced_prediction_result(self, tmp_path):
        """拡張予測結果のテスト"""
        # ダミー画像を作成
        import cv2

        from src.training.annotation_data import FrameAnnotation

        dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image_path = tmp_path / "test_frame.jpg"
        cv2.imwrite(str(image_path), dummy_image)

        frame_annotation = FrameAnnotation(
            frame_id="frame_0001",
            image_path=str(image_path),
            image_width=1920,
            image_height=1080,
            timestamp=0.0,
            tiles=[],
        )

        # 予測のテスト
        result = self.labeler.predict_frame_annotations(frame_annotation)

        assert hasattr(result, "scene_result")
        assert hasattr(result, "score_result")
        assert hasattr(result, "player_result")
        assert hasattr(result, "get_scene_context")


class TestCachedSceneDetector:
    """キャッシュ付きシーン検出器のテスト"""

    def setup_method(self):
        """テストのセットアップ"""
        self.detector = CachedSceneDetector({"cache": {"enabled": True, "size": 10}})

    def test_cache_functionality(self):
        """キャッシュ機能のテスト"""
        # 同じフレームを2回処理
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result1 = self.detector.detect_scene(frame, 0, 0.0)
        result2 = self.detector.detect_scene(frame, 1, 1.0)

        # 同じシーンタイプと信頼度が返されるべき
        assert result1.scene_type == result2.scene_type
        assert result1.confidence == result2.confidence

        # フレーム番号とタイムスタンプは異なるべき
        assert result1.frame_number != result2.frame_number
        assert result1.timestamp != result2.timestamp

    def test_cache_clear(self):
        """キャッシュクリアのテスト"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # キャッシュに追加
        self.detector.detect_scene(frame, 0, 0.0)

        # キャッシュをクリア
        self.detector.clear_cache()

        # キャッシュが空であることを確認
        assert len(self.detector._frame_hash_cache) == 0


class TestBatchProcessing:
    """バッチ処理のテスト"""

    def setup_method(self):
        """テストのセットアップ"""
        config = {"batch_processing": {"batch_size": 2, "max_workers": 2, "enable_parallel": True}}
        self.pipeline = BatchEnhancedGamePipeline(config)

    def test_batch_frame_processing(self):
        """バッチフレーム処理のテスト"""
        # テスト用フレームを作成
        frames = []
        for i in range(4):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frames.append((frame, i, float(i)))

        # バッチ処理のテスト
        results = self.pipeline.process_frames_batch(frames)

        assert len(results) == len(frames)
        for i, result in enumerate(results):
            assert result.frame_number == i

    def test_performance_stats(self):
        """パフォーマンス統計のテスト"""
        stats = self.pipeline.get_performance_stats()

        assert "batch_size" in stats
        assert "max_workers" in stats
        assert "parallel_enabled" in stats


class TestJapaneseOCR:
    """日本語OCRのテスト"""

    def test_japanese_text_extraction(self):
        """日本語テキストからの点数抽出のテスト"""
        from src.detection.score_reader import ScoreReader

        reader = ScoreReader({"use_japanese_ocr": True})

        # 日本語テキストのテストケース
        test_cases = [
            ("２５０００点", 25000),
            ("一万点", 10000),
            ("二万五千", 25000),
            ("３万２千点", 32000),
            ("１２３４５", 12345),
        ]

        for text, expected in test_cases:
            result = reader._extract_score_from_japanese_text(text)
            assert (
                result == expected or result == (expected // 100) * 100
            )  # 100点単位に丸められる可能性


class TestTypedDictUsage:
    """TypedDict使用のテスト"""

    def test_scene_metadata_type(self):
        """SceneMetadataの型テスト"""
        metadata: SceneMetadata = {
            "scene_changed": True,
            "green_ratio": 0.8,
            "hand_tiles": 14,
            "total_tiles": 34,
        }

        assert isinstance(metadata, dict)
        assert metadata["scene_changed"] is True
        assert isinstance(metadata["green_ratio"], float)
        assert isinstance(metadata["hand_tiles"], int)
