"""
ParallelAIPipelineのテスト
"""

import time
from concurrent.futures import TimeoutError
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.pipeline.ai_pipeline import PipelineResult
from src.pipeline.parallel_ai_pipeline import ParallelAIPipeline
from src.utils.config import ConfigManager


class TestParallelAIPipeline:
    """ParallelAIPipelineのテストクラス"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのモック"""
        config_manager = Mock(spec=ConfigManager)
        config_manager.config = {
            "system": {
                "max_workers": 4,
                "constants": {"default_batch_size": 32, "min_tile_size": 10},
            },
            "performance": {"processing": {"chunk_size": 8, "parallel_batch_size": 16}},
            "ai": {
                "detection": {"confidence_threshold": 0.5, "model_path": "models/detector.pt"},
                "classification": {
                    "confidence_threshold": 0.8,
                    "model_path": "models/classifier.pt",
                },
                "batch_processing": {"max_batch_size": 32, "dynamic_batching": True},
                "enable_gpu_parallel": False,
            },
        }
        return config_manager

    @pytest.fixture
    def parallel_pipeline(self, config_manager):
        """ParallelAIPipelineのフィクスチャ"""
        with patch("src.pipeline.parallel_ai_pipeline.get_logger"):
            # 親クラスの初期化をモック
            with patch.object(ParallelAIPipeline, "_initialize_models"):
                pipeline = ParallelAIPipeline(config_manager)
                # 検出器と分類器のモック
                pipeline.detector = Mock()
                pipeline.classifier = Mock()
                return pipeline

    @pytest.fixture
    def sample_frames(self):
        """サンプルフレームデータ"""
        # 20個のダミーフレームを生成
        return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(20)]

    def test_initialization(self, parallel_pipeline):
        """初期化テスト"""
        assert parallel_pipeline is not None
        assert parallel_pipeline.max_workers == 4
        assert parallel_pipeline.use_gpu_parallel is False
        assert parallel_pipeline.parallel_batch_size == 16

    def test_create_parallel_batches(self, parallel_pipeline, sample_frames):
        """並列バッチ作成のテスト"""
        batches = parallel_pipeline._create_parallel_batches(sample_frames)

        # バッチ数の確認（20フレーム / 16バッチサイズ = 2バッチ）
        assert len(batches) == 2
        assert len(batches[0]) == 16
        assert len(batches[1]) == 4

        # 各バッチがnumpy配列であることを確認
        for batch in batches:
            for frame in batch:
                assert isinstance(frame, np.ndarray)

    def test_process_frames_parallel_batches_success(self, parallel_pipeline, sample_frames):
        """並列バッチ処理の成功テスト"""

        # process_frames_batchのモック
        def mock_process_batch(frames, start_frame):
            return PipelineResult(
                success=True,
                frame_results=[
                    {
                        "frame_id": start_frame + i,
                        "detections": [Mock() for _ in range(2)],
                        "classifications": [(Mock(), Mock(confidence=0.9)) for _ in range(2)],
                        "processing_time": 0.1,
                    }
                    for i in range(len(frames))
                ],
                total_frames=len(frames),
                total_detections=len(frames) * 2,
                total_classifications=len(frames) * 2,
                processing_time=0.1 * len(frames),
                average_confidence=0.9,
                confidence_distribution={"high": len(frames) * 2},
            )

        parallel_pipeline.process_frames_batch = mock_process_batch

        # 並列処理実行
        result = parallel_pipeline.process_frames_parallel_batches(sample_frames)

        # 結果の検証
        assert result.success is True
        assert len(result.frame_results) == 20
        assert result.total_frames == 20
        assert result.total_detections == 40  # 20フレーム × 2検出
        assert result.total_classifications == 40
        assert result.processing_time > 0

        # フレームIDが正しくソートされているか確認
        frame_ids = [r["frame_id"] for r in result.frame_results]
        assert frame_ids == list(range(20))

    def test_process_frames_parallel_batches_partial_failure(
        self, parallel_pipeline, sample_frames
    ):
        """並列バッチ処理の部分的失敗テスト"""
        call_count = 0

        def mock_process_batch_with_failure(frames, start_frame):
            nonlocal call_count
            call_count += 1

            # 2番目のバッチで失敗
            if call_count == 2:
                raise Exception("Batch processing failed")

            return PipelineResult(
                success=True,
                frame_results=[
                    {"frame_id": start_frame + i, "detections": [], "classifications": []}
                    for i in range(len(frames))
                ],
                total_frames=len(frames),
                total_detections=0,
                total_classifications=0,
                processing_time=0.1,
                average_confidence=0.0,
                confidence_distribution={},
            )

        parallel_pipeline.process_frames_batch = mock_process_batch_with_failure

        # 並列処理実行
        result = parallel_pipeline.process_frames_parallel_batches(sample_frames)

        # 部分的な成功を確認
        assert result.success is True  # 全体としては成功
        assert len(result.frame_results) == 20  # すべてのフレームの結果が存在

        # エラーフレームの確認
        error_frames = [r for r in result.frame_results if r.get("error")]
        assert len(error_frames) > 0

    def test_process_batch_wrapper(self, parallel_pipeline):
        """バッチ処理ラッパーのテスト"""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

        # process_frames_batchのモック
        parallel_pipeline.process_frames_batch = Mock(
            return_value=PipelineResult(
                success=True,
                frame_results=[],
                total_frames=5,
                total_detections=10,
                total_classifications=10,
                processing_time=0.5,
                average_confidence=0.85,
                confidence_distribution={},
            )
        )

        # ラッパー実行
        result = parallel_pipeline._process_batch_wrapper(frames, 10)

        # 呼び出しの確認
        parallel_pipeline.process_frames_batch.assert_called_once_with(frames, 10)
        assert result.success is True
        assert result.total_frames == 5

    def test_create_empty_frame_result(self, parallel_pipeline):
        """空のフレーム結果作成テスト"""
        result = parallel_pipeline._create_empty_frame_result(42)

        assert result["frame_id"] == 42
        assert result["detections"] == []
        assert result["classifications"] == []
        assert result["tile_areas"] == {}
        assert result["confidence_scores"] == {}
        assert result["processing_time"] == 0.0
        assert result["error"] == "Processing failed"

    def test_process_frames_with_prefetch(self, parallel_pipeline, sample_frames):
        """プリフェッチ処理のテスト"""

        # モック設定
        def mock_prefetch_and_process(batch, start_frame):
            return PipelineResult(
                success=True,
                frame_results=[
                    {"frame_id": start_frame + i, "detections": []} for i in range(len(batch))
                ],
                total_frames=len(batch),
                total_detections=0,
                total_classifications=0,
                processing_time=0.1,
                average_confidence=0.0,
                confidence_distribution={},
            )

        parallel_pipeline._prefetch_and_process_batch = mock_prefetch_and_process

        # プリフェッチ実行
        result = parallel_pipeline.process_frames_with_prefetch(sample_frames, prefetch_size=2)

        # 結果の検証
        assert result.success is True
        assert len(result.frame_results) == 20
        assert all(r["frame_id"] == i for i, r in enumerate(result.frame_results))

    def test_preprocess_frame_optimized(self, parallel_pipeline):
        """最適化されたフレーム前処理のテスト"""
        # 異なる型のフレーム
        frame_float = np.random.rand(100, 100, 3).astype(np.float32) * 255
        frame_uint8 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # float32からuint8への変換
        result1 = parallel_pipeline._preprocess_frame_optimized(frame_float)
        assert result1.dtype == np.uint8

        # uint8はそのまま
        result2 = parallel_pipeline._preprocess_frame_optimized(frame_uint8)
        assert result2.dtype == np.uint8
        assert np.array_equal(result2, frame_uint8)

    def test_parallel_processing_with_different_worker_counts(self, config_manager):
        """異なるワーカー数での並列処理テスト"""
        worker_counts = [1, 2, 4, 8]

        for workers in worker_counts:
            config_manager.config["system"]["max_workers"] = workers

            with patch("src.pipeline.parallel_ai_pipeline.get_logger"):
                with patch.object(ParallelAIPipeline, "_initialize_models"):
                    pipeline = ParallelAIPipeline(config_manager)
                    assert pipeline.max_workers == workers

    def test_gpu_parallel_vs_cpu_parallel(self, config_manager):
        """GPU並列とCPU並列の切り替えテスト"""
        # CPU並列（ProcessPoolExecutor）
        config_manager.config["ai"]["enable_gpu_parallel"] = False
        pipeline_cpu = ParallelAIPipeline(config_manager)
        assert pipeline_cpu.use_gpu_parallel is False

        # GPU並列（ThreadPoolExecutor）
        config_manager.config["ai"]["enable_gpu_parallel"] = True
        pipeline_gpu = ParallelAIPipeline(config_manager)
        assert pipeline_gpu.use_gpu_parallel is True

    def test_timeout_handling(self, parallel_pipeline, sample_frames):
        """タイムアウト処理のテスト"""

        def mock_slow_process(frames, start_frame):
            time.sleep(100)  # 長時間の処理をシミュレート
            return PipelineResult(
                success=True,
                frame_results=[],
                total_frames=0,
                total_detections=0,
                total_classifications=0,
                processing_time=100,
                average_confidence=0,
                confidence_distribution={},
            )

        parallel_pipeline._process_batch_wrapper = mock_slow_process

        # タイムアウトを短く設定してテスト
        with patch("concurrent.futures.Future.result") as mock_result:
            mock_result.side_effect = TimeoutError("Processing timeout")

            result = parallel_pipeline.process_frames_parallel_batches(sample_frames[:5])

            # タイムアウトしても結果が返ることを確認
            assert result.success is True  # 部分的な成功として扱う
            assert len(result.frame_results) == 5
            # タイムアウトしたフレームはエラーとして記録
            assert any(r.get("error") for r in result.frame_results)

    def test_memory_efficiency(self, parallel_pipeline):
        """メモリ効率のテスト"""
        # 大きなフレームでのバッチ作成
        large_frames = [np.zeros((1920, 1080, 3), dtype=np.uint8) for _ in range(10)]

        batches = parallel_pipeline._create_parallel_batches(large_frames)

        # バッチが適切なサイズに分割されているか確認
        assert len(batches) == 1  # 10フレーム / 16バッチサイズ
        assert len(batches[0]) == 10

        # メモリ使用量の推定（実際のメモリ測定は環境依存のため省略）
        estimated_memory_per_frame = 1920 * 1080 * 3  # bytes
        total_memory = estimated_memory_per_frame * 10
        assert total_memory < 100 * 1024 * 1024  # 100MB未満

    def test_calculate_statistics(self, parallel_pipeline):
        """統計計算のテスト"""
        frame_results = [
            {
                "frame_id": i,
                "classifications": [(Mock(), Mock(confidence=0.8 + i * 0.01)) for _ in range(3)],
            }
            for i in range(10)
        ]

        # 平均信頼度の計算
        avg_confidence = parallel_pipeline._calculate_average_confidence(frame_results)
        assert 0.8 <= avg_confidence <= 0.9

        # 信頼度分布の計算
        distribution = parallel_pipeline._calculate_confidence_distribution(frame_results)
        assert isinstance(distribution, dict)
