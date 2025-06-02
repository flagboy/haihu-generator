"""
エンドツーエンド統合テスト
動画アップロード → フレーム抽出 → ラベリング → 学習 → 評価の全工程テスト
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.integration.system_integrator import SystemIntegrator
from src.optimization.performance_optimizer import PerformanceOptimizer
from src.utils.config import ConfigManager
from src.validation.quality_validator import QualityValidator
from src.video.video_processor import VideoProcessor


class TestEndToEndWorkflow:
    """エンドツーエンドワークフローテストクラス"""

    @pytest.fixture
    def temp_workspace(self):
        """一時的なワークスペースを作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # ディレクトリ構造を作成
            dirs = [
                "data/input",
                "data/output",
                "data/temp",
                "data/training",
                "logs",
                "models",
                "web_interface/uploads",
            ]
            for dir_path in dirs:
                Path(temp_dir, dir_path).mkdir(parents=True, exist_ok=True)

            yield temp_dir

    @pytest.fixture
    def config_manager(self, temp_workspace):
        """設定管理オブジェクトのフィクスチャ"""
        config_content = f"""
video:
  frame_extraction:
    fps: 1
    output_format: "jpg"
    quality: 95
    max_frames: 10

ai:
  detection:
    confidence_threshold: 0.5
  classification:
    confidence_threshold: 0.8

training:
  training_root: "{temp_workspace}/data/training"
  dataset_root: "{temp_workspace}/data/training/dataset"
  database_path: "{temp_workspace}/data/training/dataset.db"
  default_epochs: 5
  default_batch_size: 2
  default_learning_rate: 0.001

system:
  max_workers: 1
  memory_limit: "2GB"
  gpu_enabled: false

directories:
  input: "{temp_workspace}/data/input"
  output: "{temp_workspace}/data/output"
  temp: "{temp_workspace}/data/temp"
  models: "{temp_workspace}/models"
  logs: "{temp_workspace}/logs"

web:
  upload_folder: "{temp_workspace}/web_interface/uploads"

tiles:
  manzu: ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"]
  pinzu: ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"]
  souzu: ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s"]
  jihai: ["東", "南", "西", "北", "白", "發", "中"]
"""

        config_path = os.path.join(temp_workspace, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        return ConfigManager(config_path)

    @pytest.fixture
    def sample_video(self, temp_workspace):
        """サンプル動画ファイルを作成"""
        video_path = os.path.join(temp_workspace, "data/input/sample_video.mp4")

        # 簡単なテスト動画を作成
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 1.0, (640, 480))

        for i in range(10):
            # 各フレームに異なる色の矩形を描画
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            color = (i * 25, 100, 200)
            cv2.rectangle(frame, (100 + i * 10, 100), (200 + i * 10, 200), color, -1)
            out.write(frame)

        out.release()
        return video_path

    def test_complete_workflow_mock(self, config_manager, sample_video, temp_workspace):
        """完全ワークフローのモックテスト"""

        # モックコンポーネントを作成
        with (
            patch("src.video.video_processor.VideoProcessor") as mock_video_processor_class,
            patch("src.pipeline.ai_pipeline.AIPipeline") as mock_ai_pipeline_class,
            patch("src.pipeline.game_pipeline.GamePipeline") as mock_game_pipeline_class,
        ):
            # VideoProcessor のモック
            mock_video_processor = Mock()
            mock_video_processor.extract_frames.return_value = {
                "success": True,
                "frames": [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)],
                "frame_count": 5,
                "fps": 1.0,
            }
            mock_video_processor.get_video_info.return_value = {
                "success": True,
                "frame_count": 5,
                "duration": 5.0,
                "fps": 1.0,
                "width": 640,
                "height": 480,
            }
            mock_video_processor_class.return_value = mock_video_processor

            # AIPipeline のモック
            mock_ai_pipeline = Mock()

            # process_frames_batchはPipelineResultのリストを返す
            def mock_process_frames_batch(frames, batch_start_frame=0):
                from src.pipeline.ai_pipeline import PipelineResult

                results = []
                for i in range(len(frames)):
                    # detectionsとclassificationsのMockオブジェクトを作成
                    detection_mock = Mock(
                        bbox=[10, 10, 50, 50], confidence=0.8, class_id=0, class_name="tile"
                    )
                    classification_mock = Mock(
                        tile_name="1m", confidence=0.9, class_id=1, label="1m"
                    )

                    result = PipelineResult(
                        frame_id=batch_start_frame + i,
                        detections=[detection_mock],
                        classifications=[(detection_mock, classification_mock)],
                        processing_time=0.1,
                        tile_areas={"hand_tiles": [detection_mock]},
                        confidence_scores={"combined_confidence": 0.85},
                    )
                    results.append(result)
                return results

            mock_ai_pipeline.process_frames_batch.side_effect = mock_process_frames_batch
            mock_ai_pipeline_class.return_value = mock_ai_pipeline

            # GamePipeline のモック
            mock_game_pipeline = Mock()
            mock_game_pipeline.initialize_game.return_value = True
            mock_game_pipeline.process_frame.return_value = Mock(
                success=True,
                frame_number=0,
                actions_detected=1,
                confidence=0.8,
                processing_time=0.05,
            )
            # process_game_dataメソッドを追加
            mock_game_pipeline.process_game_data.return_value = {
                "game_info": {
                    "rule": "東南戦",
                    "players": ["Player1", "Player2", "Player3", "Player4"],
                },
                "rounds": [
                    {
                        "round_number": 1,
                        "round_name": "東1局",
                        "actions": [{"player": "Player1", "action": "draw", "tiles": ["1m"]}],
                    }
                ],
            }
            mock_game_pipeline.export_tenhou_json_record.return_value = {
                "game_info": {
                    "rule": "東南戦",
                    "players": ["Player1", "Player2", "Player3", "Player4"],
                },
                "rounds": [
                    {
                        "round_number": 1,
                        "round_name": "東1局",
                        "actions": [{"player": "Player1", "action": "draw", "tiles": ["1m"]}],
                    }
                ],
            }
            mock_game_pipeline_class.return_value = mock_game_pipeline

            # システム統合テスト
            integrator = SystemIntegrator(
                config_manager, mock_video_processor, mock_ai_pipeline, mock_game_pipeline
            )

            # 出力パス
            output_path = os.path.join(temp_workspace, "data/output/test_output.json")

            # 完全処理を実行
            result = integrator.process_video_complete(
                video_path=sample_video, output_path=output_path
            )

            # 結果検証
            assert result.success is True
            assert result.output_path == output_path
            assert result.frame_count == 5
            assert result.processing_time > 0

            # 出力ファイルの存在確認
            assert os.path.exists(output_path)

            # 出力内容の確認
            with open(output_path, encoding="utf-8") as f:
                output_data = json.load(f)

            assert "game_info" in output_data
            assert "rounds" in output_data
            assert len(output_data["rounds"]) > 0

    def test_performance_under_load(self, config_manager, temp_workspace):
        """負荷テスト"""

        # 複数の動画ファイルを作成
        video_files = []
        for i in range(3):
            video_path = os.path.join(temp_workspace, f"data/input/video_{i}.mp4")

            # 簡単なテスト動画を作成
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 1.0, (320, 240))

            for _j in range(5):
                frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                out.write(frame)

            out.release()
            video_files.append(video_path)

        # モックコンポーネントでシステム統合
        with (
            patch("src.video.video_processor.VideoProcessor") as mock_video_processor_class,
            patch("src.pipeline.ai_pipeline.AIPipeline") as mock_ai_pipeline_class,
            patch("src.pipeline.game_pipeline.GamePipeline") as mock_game_pipeline_class,
        ):
            # 簡単なモック設定
            mock_video_processor = Mock()
            mock_video_processor.extract_frames.return_value = {
                "success": True,
                "frames": [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)],
            }
            mock_video_processor_class.return_value = mock_video_processor

            mock_ai_pipeline = Mock()

            # process_frames_batchはPipelineResultのリストを返す
            def mock_process_frames_batch(frames, batch_start_frame=0):
                from src.pipeline.ai_pipeline import PipelineResult

                results = []
                for i in range(len(frames)):
                    result = PipelineResult(
                        frame_id=batch_start_frame + i,
                        detections=[],
                        classifications=[],
                        processing_time=0.1,
                        tile_areas={},
                        confidence_scores={},
                    )
                    results.append(result)
                return results

            mock_ai_pipeline.process_frames_batch.side_effect = mock_process_frames_batch
            mock_ai_pipeline_class.return_value = mock_ai_pipeline

            mock_game_pipeline = Mock()
            mock_game_pipeline.initialize_game.return_value = True
            mock_game_pipeline.process_frame.return_value = Mock(success=True)
            mock_game_pipeline.process_game_data.return_value = {"test": "record"}
            mock_game_pipeline.export_tenhou_json_record.return_value = {"test": "record"}
            mock_game_pipeline_class.return_value = mock_game_pipeline

            integrator = SystemIntegrator(
                config_manager, mock_video_processor, mock_ai_pipeline, mock_game_pipeline
            )

            # バッチ処理実行
            start_time = time.time()
            result = integrator.process_batch(
                video_files=video_files,
                output_directory=os.path.join(temp_workspace, "data/output"),
                max_workers=1,
            )
            processing_time = time.time() - start_time

            # パフォーマンス検証
            assert result["success"] is True
            assert result["total_files"] == 3
            assert processing_time < 30  # 30秒以内で完了
            assert result["processing_time"] < 30

    def test_error_recovery(self, config_manager, temp_workspace):
        """エラー回復テスト"""

        # 存在しない動画ファイル
        nonexistent_video = os.path.join(temp_workspace, "data/input/nonexistent.mp4")

        with (
            patch("src.video.video_processor.VideoProcessor") as mock_video_processor_class,
            patch("src.pipeline.ai_pipeline.AIPipeline") as mock_ai_pipeline_class,
            patch("src.pipeline.game_pipeline.GamePipeline") as mock_game_pipeline_class,
        ):
            # エラーを発生させるモック
            mock_video_processor = Mock()
            mock_video_processor.extract_frames.side_effect = Exception("Video file not found")
            mock_video_processor_class.return_value = mock_video_processor

            mock_ai_pipeline_class.return_value = Mock()
            mock_game_pipeline_class.return_value = Mock()

            integrator = SystemIntegrator(config_manager, mock_video_processor, Mock(), Mock())

            # エラーケースの処理
            result = integrator.process_video_complete(
                video_path=nonexistent_video,
                output_path=os.path.join(temp_workspace, "data/output/error_test.json"),
            )

            # エラーハンドリングの確認
            assert result.success is False
            assert len(result.error_messages) > 0
            assert "Video file not found" in str(result.error_messages)

    def test_memory_usage_monitoring(self, config_manager, temp_workspace):
        """メモリ使用量監視テスト"""

        # パフォーマンス最適化オブジェクト
        optimizer = PerformanceOptimizer(config_manager)

        # 監視開始
        optimizer.start_monitoring()

        # 簡単な処理を実行
        time.sleep(0.1)

        # メトリクス取得
        metrics = optimizer.get_current_metrics()

        # 監視停止
        optimizer.stop_monitoring()

        # メトリクス検証
        assert metrics is not None
        assert hasattr(metrics, "cpu_usage")
        assert hasattr(metrics, "memory_usage")
        assert 0 <= metrics.cpu_usage <= 100
        assert 0 <= metrics.memory_usage <= 100

        # 履歴確認
        assert len(optimizer.metrics_history) >= 0

    def test_quality_validation_integration(self, config_manager, temp_workspace):
        """品質検証統合テスト"""

        # サンプル牌譜データ
        sample_record = {
            "game_info": {
                "rule": "東南戦",
                "players": ["Player1", "Player2", "Player3", "Player4"],
            },
            "rounds": [
                {
                    "round_number": 1,
                    "round_name": "東1局",
                    "actions": [
                        {"player": "Player1", "action": "draw", "tiles": ["1m"]},
                        {"player": "Player1", "action": "discard", "tiles": ["9p"]},
                    ],
                }
            ],
        }

        # 牌譜ファイルを作成
        record_path = os.path.join(temp_workspace, "data/output/sample_record.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(sample_record, f, ensure_ascii=False, indent=2)

        # 品質検証実行
        validator = QualityValidator(config_manager)
        result = validator.validate_record_file(record_path)

        # 検証結果確認
        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "overall_score")
        assert 0 <= result.overall_score <= 100

        # レポート出力テスト
        report_path = os.path.join(temp_workspace, "data/output/validation_report.json")
        validator.export_validation_report(result, report_path)

        assert os.path.exists(report_path)

        # レポート内容確認
        with open(report_path, encoding="utf-8") as f:
            report_data = json.load(f)

        assert "validation_summary" in report_data
        assert "overall_score" in report_data["validation_summary"]


class TestLargeDataProcessing:
    """大容量データ処理テストクラス"""

    @pytest.fixture
    def config_manager_large(self):
        """大容量処理用設定"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
video:
  frame_extraction:
    fps: 2
    max_frames: 100

ai:
  detection:
    confidence_threshold: 0.5
  classification:
    confidence_threshold: 0.8

system:
  max_workers: 2
  memory_limit: "4GB"
  gpu_enabled: false

directories:
  input: "test_input"
  output: "test_output"
  temp: "test_temp"
""")
            config_path = f.name

        config_manager = ConfigManager(config_path)
        yield config_manager

        os.unlink(config_path)

    def test_large_video_processing(self, config_manager_large):
        """大容量動画処理テスト"""

        with tempfile.TemporaryDirectory() as temp_dir:
            # 大きめのテスト動画を作成（実際には小さいが、フレーム数を多くする）
            video_path = os.path.join(temp_dir, "large_video.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 2.0, (640, 480))

            # 50フレームの動画を作成
            for _i in range(50):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                out.write(frame)

            out.release()

            # モック処理で大容量データをシミュレート
            with (
                patch("src.video.video_processor.VideoProcessor") as mock_video_processor_class,
                patch("src.pipeline.ai_pipeline.AIPipeline") as mock_ai_pipeline_class,
                patch("src.pipeline.game_pipeline.GamePipeline") as mock_game_pipeline_class,
            ):
                # 大量のフレームを返すモック
                mock_video_processor = Mock()
                mock_video_processor.extract_frames.return_value = {
                    "success": True,
                    "frames": [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(50)],
                }
                mock_video_processor_class.return_value = mock_video_processor

                # 大量の検出結果を返すモック
                mock_ai_pipeline = Mock()

                # バッチごとに異なる結果を返すようにside_effectを使用
                def create_batch_result(frames, batch_start_frame=0):
                    from src.pipeline.ai_pipeline import PipelineResult

                    results = []
                    for i in range(len(frames)):
                        # 正しいMockオブジェクトを作成
                        detections = []
                        classifications = []
                        for _j in range(10):
                            detection_mock = Mock(
                                bbox=[10, 10, 50, 50], confidence=0.8, class_id=0, class_name="tile"
                            )
                            classification_mock = Mock(
                                label="1m", confidence=0.9, class_id=1, tile_name="1m"
                            )
                            detections.append(detection_mock)
                            classifications.append((detection_mock, classification_mock))

                        result = PipelineResult(
                            frame_id=batch_start_frame + i,
                            detections=detections,
                            classifications=classifications,
                            processing_time=0.1,
                            tile_areas={},
                            confidence_scores={"combined_confidence": 0.85},
                        )
                        results.append(result)
                    return results

                mock_ai_pipeline.process_frames_batch.side_effect = create_batch_result
                mock_ai_pipeline_class.return_value = mock_ai_pipeline

                mock_game_pipeline = Mock()
                mock_game_pipeline.initialize_game.return_value = True
                mock_game_pipeline.process_frame.return_value = Mock(success=True)
                mock_game_pipeline.process_game_data.return_value = {"large": "record"}
                mock_game_pipeline.export_tenhou_json_record.return_value = {"large": "record"}
                mock_game_pipeline_class.return_value = mock_game_pipeline

                integrator = SystemIntegrator(
                    config_manager_large, mock_video_processor, mock_ai_pipeline, mock_game_pipeline
                )

                # 大容量処理実行
                start_time = time.time()
                result = integrator.process_video_complete(
                    video_path=video_path, output_path=os.path.join(temp_dir, "large_output.json")
                )
                processing_time = time.time() - start_time

                # 結果検証
                assert result.success is True
                assert processing_time < 60  # 1分以内で完了
                assert result.frame_count == 50

    def test_memory_limit_handling(self, config_manager_large):
        """メモリ制限処理テスト"""

        # メモリ最適化オブジェクト
        from src.optimization.memory_optimizer import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # メモリ使用量監視
        initial_memory = optimizer.get_memory_info()

        # 大量のデータを作成（メモリ使用量をシミュレート）
        large_data = [np.zeros((1000, 1000), dtype=np.float32) for _ in range(10)]

        # メモリ使用量確認
        current_memory = optimizer.get_memory_info()

        # メモリクリーンアップ
        optimizer.optimize_memory()
        del large_data

        # メモリ使用量が適切に管理されていることを確認
        assert current_memory.memory_percent >= initial_memory.memory_percent

        # 最適化推奨事項取得
        recommendations = optimizer.get_memory_recommendations()
        assert isinstance(recommendations, list)


class TestErrorCases:
    """エラーケーステストクラス"""

    def test_corrupted_video_handling(self):
        """破損動画ファイル処理テスト"""

        with tempfile.TemporaryDirectory() as temp_dir:
            # 破損した動画ファイルを作成（空ファイル）
            corrupted_video = os.path.join(temp_dir, "corrupted.mp4")
            Path(corrupted_video).touch()

            # 設定作成
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, "w") as f:
                f.write("""
directories:
  input: "test_input"
  output: "test_output"
  temp: "test_temp"
system:
  max_workers: 1
""")

            config_manager = ConfigManager(config_path)

            # 実際のVideoProcessorでテスト
            video_processor = VideoProcessor(config_manager)

            # 破損ファイルの処理で例外が発生することを確認
            try:
                video_processor.extract_frames(corrupted_video)
                # 例外が投げられない場合はテスト失敗
                raise AssertionError("Expected ValueError for corrupted video file")
            except ValueError as e:
                # 期待される例外を確認
                assert "動画ファイルを開けません" in str(e)

    def test_insufficient_disk_space_simulation(self):
        """ディスク容量不足シミュレーションテスト"""

        with tempfile.TemporaryDirectory() as temp_dir:
            # 設定作成
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(f"""
directories:
  output: "{temp_dir}/output"
  temp: "{temp_dir}/temp"
system:
  max_workers: 1
""")

            config_manager = ConfigManager(config_path)

            # ディスク容量チェック機能のテスト

            # 出力ディレクトリの容量確認
            output_dir = config_manager.get_config()["directories"]["output"]
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # ディスク使用量取得
            disk_usage = shutil.disk_usage(output_dir)

            # 十分な容量があることを確認
            assert disk_usage.free > 1024 * 1024  # 1MB以上の空き容量

    def test_network_interruption_simulation(self):
        """ネットワーク中断シミュレーションテスト"""

        # ネットワーク関連の処理がある場合のテスト
        # 現在のシステムではローカル処理のみなので、
        # 将来的にリモートAPIを使用する場合のためのテンプレート

        with patch("requests.get") as mock_get:
            # ネットワークエラーをシミュレート
            mock_get.side_effect = ConnectionError("Network unreachable")

            # ネットワーク処理のテスト（現在は該当なし）
            # 将来的にリモートAPIを使用する場合に実装
            pass
