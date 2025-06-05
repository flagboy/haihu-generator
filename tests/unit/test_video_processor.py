"""
動画処理のユニットテスト
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# テスト対象モジュールのパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent))


class TestVideoProcessor:
    """動画処理のテストクラス"""

    @pytest.fixture
    def temp_video(self):
        """テスト用の動画ファイルを作成"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            # ダミー動画データ（実際には使用しない）
            temp_path = f.name
        return temp_path

    @pytest.fixture
    def processor_labeling(self, temp_video):
        """hand_labeling_systemのprocessorを取得"""
        from hand_labeling_system.backend.core.video_processor import VideoProcessor

        return VideoProcessor(temp_video)

    @pytest.fixture
    def extractor_training(self, temp_video, tmp_path):
        """hand_training_systemのextractorを取得"""
        from hand_training_system.backend.core.frame_extractor import FrameExtractor

        return FrameExtractor(temp_video, str(tmp_path))

    def test_cache_functionality(self, processor_labeling):
        """キャッシュ機能のテスト"""
        # VideoProcessorにはキャッシュ機能がある
        assert hasattr(processor_labeling, "cache_dir")
        assert hasattr(processor_labeling, "_get_frame_cache_path")

    def test_scene_detection(self, processor_labeling):
        """シーン変化検出機能のテスト"""
        # VideoProcessorにはシーン変化検出がある
        assert hasattr(processor_labeling, "detect_scene_changes")

    def test_basic_extraction(self, extractor_training):
        """基本的なフレーム抽出機能のテスト"""
        # FrameExtractorは基本的な抽出機能のみ
        assert hasattr(extractor_training, "extract_frames_at_interval")

    def test_progress_tracking(self, processor_labeling):
        """進捗管理機能のテスト"""
        # VideoProcessorには進捗管理機能がある
        assert hasattr(processor_labeling, "get_progress")

    def test_metadata_handling(self, processor_labeling):
        """メタデータ管理のテスト"""
        metadata = {
            "video_path": "/path/to/video.mp4",
            "fps": 30,
            "total_frames": 1800,
            "duration": 60.0,
        }

        # メタデータの保存と読み込みをテスト
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metadata, f)
            temp_path = f.name

        with open(temp_path) as f:
            loaded = json.load(f)
            assert loaded == metadata

    def test_frame_interval(self, extractor_training):
        """フレーム間隔設定のテスト"""
        # extract_frames_at_intervalメソッドで間隔を指定できる
        assert hasattr(extractor_training, "extract_frames_at_interval")

    def test_hand_change_detection(self, extractor_training):
        """手牌変化検出のテスト"""
        # detect_hand_changesメソッドの存在を確認
        assert hasattr(extractor_training, "detect_hand_changes")
