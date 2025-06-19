"""
動画処理モジュールのテスト
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.utils.config import ConfigManager
from src.video.video_processor import VideoProcessor


class TestVideoProcessor:
    """VideoProcessorクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.config_manager = ConfigManager()
        self.processor = VideoProcessor(self.config_manager)

    def test_initialization(self):
        """初期化のテスト"""
        assert self.processor.config_manager is not None
        assert self.processor.fps > 0
        assert self.processor.target_width > 0
        assert self.processor.target_height > 0

    def test_resize_frame(self):
        """フレームリサイズのテスト"""
        # テスト用フレームを作成（640x480）
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 1920x1080にリサイズ
        resized = self.processor.resize_frame(test_frame, 1920, 1080)

        # サイズが正しいことを確認
        assert resized.shape == (1080, 1920, 3)

        # 元のアスペクト比が保持されていることを確認（パディング込み）
        assert resized.dtype == np.uint8

    def test_normalize_frame(self):
        """フレーム正規化のテスト"""
        # テスト用フレームを作成
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # 正規化実行
        normalized = self.processor.normalize_frame(test_frame)

        # サイズと型が保持されていることを確認
        assert normalized.shape == test_frame.shape
        assert normalized.dtype == np.uint8

    def test_preprocess_frame(self):
        """フレーム前処理のテスト"""
        # テスト用フレームを作成
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 前処理実行
        processed = self.processor.preprocess_frame(test_frame)

        # 目標サイズになっていることを確認
        assert processed.shape == (self.processor.target_height, self.processor.target_width, 3)
        assert processed.dtype == np.uint8

    def test_is_valid_frame(self):
        """フレーム有効性チェックのテスト"""
        # 有効なフレーム
        valid_frame = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        assert self.processor._is_valid_frame(valid_frame)

        # 無効なフレーム（None）
        assert not self.processor._is_valid_frame(None)

        # 無効なフレーム（空配列）
        empty_frame = np.array([])
        assert not self.processor._is_valid_frame(empty_frame)

        # 無効なフレーム（極端に暗い）
        dark_frame = np.full((100, 100, 3), 10, dtype=np.uint8)
        assert not self.processor._is_valid_frame(dark_frame)

        # 無効なフレーム（極端に明るい）
        bright_frame = np.full((100, 100, 3), 250, dtype=np.uint8)
        assert not self.processor._is_valid_frame(bright_frame)

    @patch("cv2.VideoCapture")
    def test_get_video_info(self, mock_video_capture):
        """動画情報取得のテスト"""
        # モックの設定
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap

        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # 動画情報を取得
            info = self.processor.get_video_info(temp_path)

            # 情報が正しく取得されていることを確認
            assert info["width"] == 1920
            assert info["height"] == 1080
            assert info["fps"] == 30.0
            assert info["frame_count"] == 900
            assert info["duration"] == 30.0  # 900フレーム / 30fps = 30秒

        finally:
            Path(temp_path).unlink()

    def test_get_video_info_file_not_found(self):
        """存在しない動画ファイルのテスト"""
        with pytest.raises(FileNotFoundError):
            self.processor.get_video_info("non_existent_video.mp4")

    @patch("cv2.VideoCapture")
    def test_get_video_info_cannot_open(self, mock_video_capture):
        """開けない動画ファイルのテスト"""
        # モックの設定（開けない）
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with pytest.raises(ValueError):
                self.processor.get_video_info(temp_path)

        finally:
            Path(temp_path).unlink()

    @patch("cv2.VideoCapture")
    def test_detect_scene_changes(self, mock_video_capture):
        """シーン変更検出のテスト"""
        # モックの設定
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # FPS

        # フレームデータを準備（シーン変更をシミュレート）
        frame1 = np.full((100, 100, 3), 50, dtype=np.uint8)  # 暗いフレーム
        frame2 = np.full((100, 100, 3), 200, dtype=np.uint8)  # 明るいフレーム（シーン変更）
        frame3 = np.full((100, 100, 3), 60, dtype=np.uint8)  # 再び暗いフレーム

        # read()の戻り値を設定
        mock_cap.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (True, frame3),
            (False, None),  # 終了
        ]
        mock_video_capture.return_value = mock_cap

        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # シーン変更検出を実行
            scene_changes = self.processor.detect_scene_changes(temp_path, threshold=0.1)

            # シーン変更が検出されることを確認
            assert len(scene_changes) > 0

        finally:
            Path(temp_path).unlink()

    def test_filter_relevant_frames(self):
        """関連フレームフィルタリングのテスト"""
        # テスト用の画像ファイルを作成
        temp_dir = Path(tempfile.mkdtemp())
        frame_paths = []

        try:
            # 有効なフレームを作成
            for i in range(3):
                frame = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
                frame_path = temp_dir / f"frame_{i}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))

            # 無効なフレーム（極端に暗い）を作成
            dark_frame = np.full((100, 100, 3), 10, dtype=np.uint8)
            dark_frame_path = temp_dir / "dark_frame.jpg"
            cv2.imwrite(str(dark_frame_path), dark_frame)
            frame_paths.append(str(dark_frame_path))

            # フィルタリング実行
            filtered_frames = self.processor.filter_relevant_frames(frame_paths)

            # 有効なフレームのみが残ることを確認
            assert len(filtered_frames) == 3  # 暗いフレームは除外される

        finally:
            # 一時ファイルを削除
            for frame_path in frame_paths:
                Path(frame_path).unlink(missing_ok=True)
            temp_dir.rmdir()

    def test_custom_config(self):
        """カスタム設定でのテスト"""
        # カスタム設定を作成
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
video:
  frame_extraction:
    fps: 2
    output_format: "png"
    quality: 90
  preprocessing:
    target_width: 1280
    target_height: 720
    normalize: false
    denoise: false
""")
            temp_config_path = f.name

        try:
            # カスタム設定でVideoProcessorを初期化
            custom_config = ConfigManager(temp_config_path)
            custom_processor = VideoProcessor(custom_config)

            # カスタム設定が反映されていることを確認
            assert custom_processor.fps == 2
            assert custom_processor.output_format == "png"
            assert custom_processor.quality == 90
            assert custom_processor.target_width == 1280
            assert custom_processor.target_height == 720
            assert custom_processor.normalize is False
            assert custom_processor.denoise is False

        finally:
            Path(temp_config_path).unlink()
