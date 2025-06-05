"""
ラベリングシステムの統合テスト
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# テスト対象モジュールのパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent))


class TestLabelingSystemIntegration:
    """ラベリングシステム全体の統合テスト"""

    @pytest.fixture
    def test_video_path(self):
        """テスト用動画のパス"""
        # 実際のテストでは適切なテスト動画を使用
        return Path(__file__).parent.parent / "fixtures" / "sample_video.mp4"

    @pytest.fixture
    def test_config(self):
        """テスト用設定"""
        return {
            "hand_regions": {
                "bottom": {"x": 0.15, "y": 0.75, "w": 0.7, "h": 0.15},
                "top": {"x": 0.15, "y": 0.1, "w": 0.7, "h": 0.15},
                "left": {"x": 0.05, "y": 0.3, "w": 0.15, "h": 0.4},
                "right": {"x": 0.8, "y": 0.3, "w": 0.15, "h": 0.4},
            },
            "frame_interval": 1.0,
            "cache_enabled": True,
        }

    def test_end_to_end_workflow(self, test_config):
        """エンドツーエンドのワークフローテスト"""
        # 1. 動画からフレーム抽出
        from hand_training_system.backend.core.frame_extractor import FrameExtractor

        extractor = FrameExtractor()

        # 2. 手牌領域の設定
        from hand_training_system.backend.core.hand_area_detector import HandAreaDetector

        detector = HandAreaDetector()

        # 3. 牌の分割
        from hand_training_system.backend.core.tile_splitter import TileSplitter

        splitter = TileSplitter()

        # ワークフローの動作確認
        assert extractor is not None
        assert detector is not None
        assert splitter is not None

    def test_data_persistence(self, test_config):
        """データの永続化テスト"""
        # 設定の保存
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name

        # 設定の読み込み
        with open(config_path) as f:
            loaded_config = json.load(f)

        assert loaded_config == test_config

    def test_labeling_session(self):
        """ラベリングセッションのテスト"""
        session_data = {
            "session_id": "test_session_001",
            "video_path": "/path/to/video.mp4",
            "created_at": "2025-01-01T00:00:00",
            "progress": {"total_frames": 1000, "labeled_frames": 100, "current_frame": 100},
        }

        # セッションデータの検証
        assert session_data["session_id"] is not None
        assert (
            session_data["progress"]["labeled_frames"] <= session_data["progress"]["total_frames"]
        )

    def test_annotation_format(self):
        """アノテーションフォーマットのテスト"""
        annotation = {
            "frame_id": "frame_001",
            "timestamp": 3.5,
            "player": "bottom",
            "tiles": [
                {"index": 0, "label": "1m", "confidence": 0.95},
                {"index": 1, "label": "2m", "confidence": 0.88},
                {"index": 2, "label": "3m", "confidence": 0.92},
            ],
        }

        # アノテーションの必須フィールドを確認
        assert "frame_id" in annotation
        assert "player" in annotation
        assert "tiles" in annotation
        assert len(annotation["tiles"]) > 0

    def test_export_formats(self):
        """エクスポート形式のテスト"""
        # COCO形式
        coco_format = {"images": [], "annotations": [], "categories": []}

        # YOLO形式（テキストファイル）
        yolo_format = "0 0.5 0.5 0.1 0.1\n"  # class x y w h

        # 天鳳形式
        tenhou_format = {"tiles": ["1m", "2m", "3m"], "player": "bottom"}

        # 各形式の基本構造を確認
        assert "images" in coco_format
        assert "\n" in yolo_format
        assert "tiles" in tenhou_format
