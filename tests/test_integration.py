"""
統合テスト
システム全体の統合テストを実行
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.integration.system_integrator import IntegrationResult, SystemIntegrator
from src.optimization.performance_optimizer import PerformanceOptimizer
from src.utils.config import ConfigManager
from src.validation.quality_validator import QualityValidator


class TestSystemIntegration:
    """システム統合テストクラス"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのフィクスチャ"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
video:
  frame_extraction:
    fps: 1
    output_format: "jpg"
    quality: 95

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
  models: "test_models"
  logs: "test_logs"
""")
            config_path = f.name

        config_manager = ConfigManager(config_path)
        yield config_manager

        # クリーンアップ
        os.unlink(config_path)

    @pytest.fixture
    def mock_components(self):
        """モックコンポーネントのフィクスチャ"""
        # VideoProcessor のモック
        video_processor = Mock()
        video_processor.extract_frames.return_value = {
            "success": True,
            "frames": [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)],
        }
        video_processor.get_video_info.return_value = {
            "success": True,
            "frame_count": 5,
            "duration": 5.0,
            "fps": 1.0,
        }

        # AIPipeline のモック
        ai_pipeline = Mock()

        def mock_ai_processing(frames, batch_start_frame=0):
            # PipelineResultのリストを返す
            from src.pipeline.ai_pipeline import PipelineResult

            results = []
            for i in range(len(frames)):
                result = PipelineResult(
                    frame_id=batch_start_frame + i,
                    detections=[
                        Mock(bbox=[10, 10, 50, 50], confidence=0.8, class_id=0, class_name="tile")
                    ],
                    classifications=[
                        (
                            Mock(
                                bbox=[10, 10, 50, 50], confidence=0.8, class_id=0, class_name="tile"
                            ),
                            Mock(label="1m", confidence=0.9, class_id=1),
                        )
                    ],
                    processing_time=0.1,
                    tile_areas={
                        "hand_tiles": [
                            Mock(
                                bbox=[10, 10, 50, 50], confidence=0.8, class_id=0, class_name="tile"
                            )
                        ]
                    },
                    confidence_scores={"combined_confidence": 0.85},
                )
                results.append(result)
            return results

        ai_pipeline.process_frames_batch.side_effect = mock_ai_processing

        # GamePipeline のモック
        game_pipeline = Mock()
        game_pipeline.initialize_game.return_value = True
        game_pipeline.process_frame.return_value = Mock(
            success=True, frame_number=0, actions_detected=1, confidence=0.8, processing_time=0.05
        )
        game_pipeline.process_game_data.return_value = Mock(
            get_statistics=lambda: {"rounds": 1, "actions": 10},
            to_tenhou_format=lambda: {
                "title": "天鳳サンプル牌譜",
                "players": [
                    {"name": "プレイヤー1", "score": 25000},
                    {"name": "プレイヤー2", "score": 25000},
                    {"name": "プレイヤー3", "score": 25000},
                    {"name": "プレイヤー4", "score": 25000},
                ],
                "rounds": [{"round_number": 0, "actions": [["T0", "1m"]]}],
            },
        )
        game_pipeline.export_tenhou_json_record.return_value = {
            "title": "天鳳サンプル牌譜",
            "players": [
                {"name": "プレイヤー1", "score": 25000},
                {"name": "プレイヤー2", "score": 25000},
                {"name": "プレイヤー3", "score": 25000},
                {"name": "プレイヤー4", "score": 25000},
            ],
            "rounds": [{"round_number": 0, "actions": [["T0", "1m"]]}],
        }
        game_pipeline.export_game_record.return_value = '{"game": "test_record"}'

        return video_processor, ai_pipeline, game_pipeline

    @pytest.fixture
    def system_integrator(self, config_manager, mock_components):
        """システム統合オブジェクトのフィクスチャ"""
        video_processor, ai_pipeline, game_pipeline = mock_components
        return SystemIntegrator(config_manager, video_processor, ai_pipeline, game_pipeline)

    def test_system_integrator_initialization(self, system_integrator):
        """システム統合オブジェクトの初期化テスト"""
        assert system_integrator is not None
        assert system_integrator.video_processor is not None
        assert system_integrator.ai_pipeline is not None
        assert system_integrator.game_pipeline is not None
        assert system_integrator.integration_config is not None
        # リファクタリングされたコンポーネントの確認
        assert system_integrator.orchestrator is not None
        assert system_integrator.result_processor is not None
        assert system_integrator.statistics_collector is not None

    def test_complete_video_processing(self, system_integrator):
        """完全な動画処理テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テスト用の動画ファイル（ダミー）
            video_path = os.path.join(temp_dir, "test_video.mp4")
            Path(video_path).touch()

            # 出力パス
            output_path = os.path.join(temp_dir, "test_output.json")

            # 処理実行（天鳳JSON形式専用）
            result = system_integrator.process_video_complete(
                video_path=video_path, output_path=output_path
            )

            # 結果検証
            assert isinstance(result, IntegrationResult)
            assert result.success is True
            assert result.output_path == output_path
            assert result.frame_count == 5
            assert result.detection_count > 0
            assert result.classification_count > 0
            assert result.processing_time > 0

            # 出力ファイルの存在確認
            assert os.path.exists(output_path)

    def test_batch_processing(self, system_integrator):
        """バッチ処理テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テスト用の動画ファイル（複数）
            video_files = []
            for i in range(3):
                video_path = os.path.join(temp_dir, f"test_video_{i}.mp4")
                Path(video_path).touch()
                video_files.append(video_path)

            # 出力ディレクトリ
            output_dir = os.path.join(temp_dir, "output")

            # バッチ処理実行（天鳳JSON形式専用）
            result = system_integrator.process_batch(
                video_files=video_files, output_directory=output_dir, max_workers=2
            )

            # 結果検証
            assert result["success"] is True
            assert result["total_files"] == 3
            assert result["successful_count"] >= 0
            assert result["processing_time"] > 0
            assert len(result["results"]) == 3

    def test_progress_tracking(self, system_integrator):
        """進捗追跡テスト"""
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress)

        system_integrator.add_progress_callback(progress_callback)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "test_video.mp4")
            Path(video_path).touch()
            output_path = os.path.join(temp_dir, "test_output.json")

            # 処理実行
            system_integrator.process_video_complete(video_path=video_path, output_path=output_path)

            # 進捗更新の確認
            assert len(progress_updates) > 0
            assert any(p.progress_percentage == 100.0 for p in progress_updates)

    def test_error_handling(self, system_integrator):
        """エラーハンドリングテスト"""
        # 存在しない動画ファイル
        result = system_integrator.process_video_complete(
            video_path="nonexistent_video.mp4", output_path="output.json"
        )

        # エラー結果の確認
        assert result.success is False
        assert len(result.error_messages) > 0

    def test_system_info(self, system_integrator):
        """システム情報取得テスト"""
        system_info = system_integrator.get_system_info()

        assert "cpu_count" in system_info
        assert "memory_total_gb" in system_info
        assert "integration_config" in system_info
        assert "component_status" in system_info


class TestPerformanceOptimization:
    """パフォーマンス最適化テストクラス"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのフィクスチャ"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
system:
  max_workers: 4
  memory_limit: "8GB"
  gpu_enabled: false
""")
            config_path = f.name

        config_manager = ConfigManager(config_path)
        yield config_manager

        os.unlink(config_path)

    @pytest.fixture
    def performance_optimizer(self, config_manager):
        """パフォーマンス最適化オブジェクトのフィクスチャ"""
        return PerformanceOptimizer(config_manager)

    def test_performance_optimizer_initialization(self, performance_optimizer):
        """パフォーマンス最適化オブジェクトの初期化テスト"""
        assert performance_optimizer is not None
        assert performance_optimizer.optimization_config is not None

    def test_get_current_metrics(self, performance_optimizer):
        """現在のメトリクス取得テスト"""
        metrics = performance_optimizer.get_current_metrics()

        assert metrics is not None
        assert hasattr(metrics, "cpu_usage")
        assert hasattr(metrics, "memory_usage")
        assert hasattr(metrics, "timestamp")
        assert 0 <= metrics.cpu_usage <= 100
        assert 0 <= metrics.memory_usage <= 100

    def test_system_optimization(self, performance_optimizer):
        """システム最適化テスト"""
        result = performance_optimizer.optimize_system()

        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "optimization_type")
        assert hasattr(result, "before_metrics")
        assert hasattr(result, "after_metrics")

    def test_monitoring(self, performance_optimizer):
        """監視機能テスト"""
        # 監視開始
        performance_optimizer.start_monitoring()
        assert performance_optimizer.monitoring_active is True

        # 少し待機
        import time

        time.sleep(0.1)

        # 監視停止
        performance_optimizer.stop_monitoring()
        assert performance_optimizer.monitoring_active is False

        # メトリクス履歴の確認
        assert len(performance_optimizer.metrics_history) >= 0

    def test_batch_size_optimization(self, performance_optimizer):
        """バッチサイズ最適化テスト"""
        # 正常ケース
        optimized_size = performance_optimizer.optimize_batch_size(
            current_batch_size=8, processing_time=0.5, memory_usage=50.0
        )
        assert isinstance(optimized_size, int)
        assert optimized_size > 0

        # メモリ使用率が高い場合
        optimized_size = performance_optimizer.optimize_batch_size(
            current_batch_size=16, processing_time=1.0, memory_usage=90.0
        )
        assert optimized_size <= 16  # バッチサイズが減少するはず

    def test_recommendations(self, performance_optimizer):
        """推奨事項取得テスト"""
        recommendations = performance_optimizer.get_optimization_recommendations()

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestQualityValidation:
    """品質検証テストクラス"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのフィクスチャ"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
tiles:
  manzu: ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"]
  pinzu: ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"]
  souzu: ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s"]
  jihai: ["東", "南", "西", "北", "白", "發", "中"]
""")
            config_path = f.name

        config_manager = ConfigManager(config_path)
        yield config_manager

        os.unlink(config_path)

    @pytest.fixture
    def quality_validator(self, config_manager):
        """品質検証オブジェクトのフィクスチャ"""
        return QualityValidator(config_manager)

    @pytest.fixture
    def sample_record_data(self):
        """サンプル牌譜データのフィクスチャ"""
        return {
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

    def test_quality_validator_initialization(self, quality_validator):
        """品質検証オブジェクトの初期化テスト"""
        assert quality_validator is not None
        assert quality_validator.validation_config is not None
        assert quality_validator.validation_rules is not None

    def test_record_data_validation(self, quality_validator, sample_record_data):
        """牌譜データ検証テスト"""
        result = quality_validator.validate_record_data(sample_record_data)

        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "overall_score")
        assert hasattr(result, "category_scores")
        assert hasattr(result, "issues")
        assert hasattr(result, "recommendations")

        # スコアの範囲チェック
        assert 0 <= result.overall_score <= 100
        for score in result.category_scores.values():
            assert 0 <= score <= 100

    def test_record_file_validation(self, quality_validator, sample_record_data):
        """牌譜ファイル検証テスト"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_record_data, f, ensure_ascii=False)
            record_path = f.name

        try:
            result = quality_validator.validate_record_file(record_path)

            assert result is not None
            assert hasattr(result, "success")
            assert hasattr(result, "overall_score")

        finally:
            os.unlink(record_path)

    def test_invalid_record_validation(self, quality_validator):
        """無効な牌譜の検証テスト"""
        invalid_data = {"invalid_field": "invalid_value"}

        result = quality_validator.validate_record_data(invalid_data)

        # 無効なデータでもエラーにならず、低いスコアが返されることを確認
        assert result is not None
        assert result.overall_score < 50  # 低いスコア
        assert len(result.issues) > 0  # 問題が検出される

    def test_validation_report_export(self, quality_validator, sample_record_data):
        """検証レポートエクスポートテスト"""
        result = quality_validator.validate_record_data(sample_record_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report_path = f.name

        try:
            quality_validator.export_validation_report(result, report_path)

            # レポートファイルの存在確認
            assert os.path.exists(report_path)

            # レポート内容の確認
            with open(report_path, encoding="utf-8") as f:
                report_data = json.load(f)

            assert "validation_summary" in report_data
            assert "issues" in report_data
            assert "recommendations" in report_data

        finally:
            if os.path.exists(report_path):
                os.unlink(report_path)


class TestEndToEndIntegration:
    """エンドツーエンド統合テストクラス"""

    def test_full_pipeline_mock(self):
        """完全パイプラインのモックテスト"""
        # 設定ファイル作成
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
video:
  frame_extraction:
    fps: 1
ai:
  detection:
    confidence_threshold: 0.5
system:
  max_workers: 1
directories:
  input: "test_input"
  output: "test_output"
  temp: "test_temp"
""")
            config_path = f.name

        try:
            # コンポーネント初期化
            config_manager = ConfigManager(config_path)

            # モックコンポーネント作成
            video_processor = Mock()
            video_processor.extract_frames.return_value = {
                "success": True,
                "frames": [np.zeros((100, 100, 3), dtype=np.uint8)],
            }

            ai_pipeline = Mock()

            def mock_process_frames(frames, batch_start_frame=0):
                # PipelineResultのリストを返す
                from src.pipeline.ai_pipeline import PipelineResult

                results = []
                for i in range(len(frames)):
                    result = PipelineResult(
                        frame_id=batch_start_frame + i,
                        detections=[],
                        classifications=[],
                        processing_time=0.1,
                        tile_areas={},
                        confidence_scores={"combined_confidence": 0.8},
                    )
                    results.append(result)
                return results

            ai_pipeline.process_frames_batch.side_effect = mock_process_frames

            game_pipeline = Mock()
            game_pipeline.initialize_game.return_value = True
            game_pipeline.process_frame.return_value = Mock(success=True)
            game_pipeline.export_tenhou_json_record.return_value = {"test": "record"}
            game_pipeline.export_game_record.return_value = '{"test": "record"}'

            # システム統合
            integrator = SystemIntegrator(
                config_manager, video_processor, ai_pipeline, game_pipeline
            )

            # パフォーマンス最適化
            optimizer = PerformanceOptimizer(config_manager)

            # 品質検証
            validator = QualityValidator(config_manager)

            # 各コンポーネントが正常に初期化されることを確認
            assert integrator is not None
            assert optimizer is not None
            assert validator is not None

            # 簡単な処理テスト
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, "test.mp4")
                Path(video_path).touch()
                output_path = os.path.join(temp_dir, "output.json")

                result = integrator.process_video_complete(video_path, output_path)
                assert result.success is True

        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
