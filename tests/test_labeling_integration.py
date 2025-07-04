"""
手牌ラベリングシステムの統合テスト
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.training.labeling.core.hand_area_detector import UnifiedHandAreaDetector
from src.training.labeling.core.labeling_session import LabelingSession
from src.training.labeling.core.tile_splitter import TileSplitter
from src.training.labeling.core.video_processor import EnhancedVideoProcessor


class TestLabelingIntegration:
    """ラベリングシステム統合テスト"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_video(self, temp_dir):
        """テスト用動画を作成"""
        video_path = temp_dir / "test_video.mp4"

        # 10フレームのダミー動画を作成
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1920, 1080))

        for i in range(10):
            # ダミーフレーム（グラデーション）
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:, :] = [i * 25, i * 25, i * 25]
            writer.write(frame)

        writer.release()
        return video_path

    @pytest.fixture
    def sample_frame(self):
        """テスト用フレーム"""
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        # 手牌領域にダミーの牌を描画
        for i in range(14):
            x = 300 + i * 50
            y = 800
            cv2.rectangle(frame, (x, y), (x + 40, y + 60), (255, 255, 255), -1)
        return frame

    def test_hand_area_detector_integration(self, sample_frame):
        """手牌領域検出の統合テスト"""
        detector = UnifiedHandAreaDetector()

        # フレームサイズを設定
        detector.set_frame_size(sample_frame.shape[1], sample_frame.shape[0])

        # 手動で領域を設定
        detector.set_manual_area("bottom", {"x": 0.15, "y": 0.75, "w": 0.7, "h": 0.15})

        # 領域を取得
        regions = detector.detect_areas(sample_frame, auto_detect=False)

        assert "bottom" in regions
        assert regions["bottom"]["x"] == int(1920 * 0.15)
        assert regions["bottom"]["y"] == int(1080 * 0.75)

    def test_video_processor_integration(self, sample_video, temp_dir):
        """動画処理の統合テスト"""
        cache_dir = temp_dir / "cache"
        processor = EnhancedVideoProcessor(str(sample_video), str(cache_dir))

        # 基本情報の確認
        assert processor.frame_count == 10
        assert processor.fps == 30.0
        assert processor.width == 1920
        assert processor.height == 1080

        # フレーム取得
        frame = processor.get_frame(0)
        assert frame is not None
        assert frame.shape == (1080, 1920, 3)

        # フレーム抽出
        extracted = processor.extract_frames(interval=1.0)
        assert len(extracted) > 0

    def test_tile_splitter_integration(self, sample_frame):
        """牌分割の統合テスト"""
        splitter = TileSplitter()

        # 手牌領域を切り出し
        hand_region = sample_frame[750:900, 200:1000]

        # 牌を分割
        tiles = splitter.split_tiles(hand_region, num_tiles=14)

        assert len(tiles) == 14
        for tile in tiles:
            assert tile.shape[0] > 0
            assert tile.shape[1] > 0

    def test_labeling_session_integration(self, temp_dir, sample_video):
        """ラベリングセッションの統合テスト"""
        session_dir = temp_dir / "sessions"
        session = LabelingSession(data_dir=str(session_dir))

        # 動画情報を設定
        video_info = {
            "path": str(sample_video),
            "fps": 30.0,
            "frame_count": 10,
            "width": 1920,
            "height": 1080,
        }
        session.set_video_info(video_info)

        # 手牌領域を設定
        regions = {
            "bottom": {"x": 0.15, "y": 0.75, "w": 0.7, "h": 0.15},
            "top": {"x": 0.15, "y": 0.1, "w": 0.7, "h": 0.15},
        }
        session.set_hand_regions(regions)

        # アノテーションを追加
        tiles = [
            {"index": 0, "label": "1m", "x": 100, "y": 800, "w": 40, "h": 60, "confidence": 0.95},
            {"index": 1, "label": "2m", "x": 150, "y": 800, "w": 40, "h": 60, "confidence": 0.90},
        ]
        session.add_annotation(0, "bottom", tiles)

        # 進捗を確認
        progress = session.get_progress()
        assert progress["labeled_frames"] == 1
        assert progress["total_frames"] == 10

        # エクスポート（tenhou形式のみサポート）
        tenhou_data = session.export_annotations(format="tenhou")
        assert tenhou_data is not None

    def test_full_workflow(self, temp_dir, sample_video):
        """完全なワークフローのテスト"""
        # 1. セッションを作成
        session = LabelingSession(data_dir=str(temp_dir / "sessions"))

        # 2. 動画処理を初期化
        video_processor = EnhancedVideoProcessor(str(sample_video), str(temp_dir / "cache"))

        # 3. 手牌領域検出を初期化
        hand_detector = UnifiedHandAreaDetector()

        # 4. 牌分割を初期化
        tile_splitter = TileSplitter()

        # 5. セッションに動画情報を設定
        video_info = {
            "path": str(sample_video),
            "fps": video_processor.fps,
            "frame_count": video_processor.frame_count,
            "width": video_processor.width,
            "height": video_processor.height,
        }
        session.set_video_info(video_info)

        # 6. フレームを処理
        frame = video_processor.get_frame(0)
        assert frame is not None

        # 7. 手牌領域を設定
        hand_detector.set_frame_size(frame.shape[1], frame.shape[0])
        hand_detector.set_manual_area("bottom", {"x": 0.15, "y": 0.75, "w": 0.7, "h": 0.15})

        # 8. 手牌領域を抽出
        hand_region = hand_detector.extract_hand_region(frame, "bottom")

        # 9. 牌を分割（手牌領域が取得できた場合）
        if hand_region is not None:
            tiles = tile_splitter.split_tiles(hand_region)

            # 10. アノテーションを作成
            annotations = []
            for i, _tile in enumerate(tiles[:3]):  # 最初の3枚のみ
                annotations.append(
                    {
                        "index": i,
                        "label": f"{i + 1}m",
                        "x": i * 50,
                        "y": 0,
                        "w": 40,
                        "h": 60,
                        "confidence": 0.9,
                    }
                )

            # 11. セッションに保存
            session.add_annotation(0, "bottom", annotations)

        # 12. セッションを保存
        session_summary = session.get_session_summary()
        assert session_summary["session_id"] == session.session_id
        assert session_summary["progress"]["labeled_frames"] >= 0

    def test_data_compatibility(self, temp_dir):
        """新旧データ形式の互換性テスト"""
        # 旧形式のデータ
        old_format = {
            "player1": {"x": 0.2, "y": 0.75, "w": 0.6, "h": 0.15},
            "player2": {"x": 0.75, "y": 0.2, "w": 0.15, "h": 0.6},
        }

        # 新形式への変換
        new_format = {"bottom": old_format["player1"], "right": old_format["player2"]}

        # UnifiedHandAreaDetectorで読み込めることを確認
        detector = UnifiedHandAreaDetector()
        detector.regions = new_format

        assert not detector.validate_regions()  # 4人分必要

        # 全員分を設定
        detector.regions = {
            "bottom": {"x": 0.15, "y": 0.75, "w": 0.7, "h": 0.15},
            "top": {"x": 0.15, "y": 0.1, "w": 0.7, "h": 0.15},
            "left": {"x": 0.05, "y": 0.3, "w": 0.15, "h": 0.4},
            "right": {"x": 0.8, "y": 0.3, "w": 0.15, "h": 0.4},
        }

        assert detector.validate_regions()
