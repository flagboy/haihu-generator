#!/usr/bin/env python3
"""
システムベンチマークテスト
パフォーマンス測定とシステム評価
"""

import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optimization.memory_optimizer import MemoryOptimizer
from src.optimization.performance_optimizer import PerformanceOptimizer
from src.utils.config import ConfigManager
from src.utils.logger import get_logger


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""

    test_name: str
    success: bool
    processing_time: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_message: str = ""
    additional_metrics: dict[str, Any] = None


class SystemBenchmark:
    """システムベンチマーククラス"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化

        Args:
            config_path: 設定ファイルパス
        """
        self.config_manager = ConfigManager(config_path)
        self.logger = get_logger(__name__)

        # ベンチマーク結果
        self.results: list[BenchmarkResult] = []

        # システム情報
        self.system_info = self._get_system_info()

        # 一時ディレクトリ
        self.temp_dir = None

    def _get_system_info(self) -> dict[str, Any]:
        """システム情報を取得"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": sys.platform,
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat(),
        }

    def setup_benchmark_environment(self) -> bool:
        """ベンチマーク環境をセットアップ"""
        try:
            self.logger.info("Setting up benchmark environment...")

            # 一時ディレクトリ作成
            self.temp_dir = tempfile.mkdtemp(prefix="benchmark_")
            self.logger.info(f"Benchmark directory: {self.temp_dir}")

            # 必要なディレクトリ構造を作成
            dirs = ["input", "output", "temp", "models"]
            for dir_name in dirs:
                Path(self.temp_dir, dir_name).mkdir(parents=True, exist_ok=True)

            # ベンチマーク用設定を作成
            self._create_benchmark_config()

            self.logger.info("Benchmark environment setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup benchmark environment: {e}")
            return False

    def _create_benchmark_config(self):
        """ベンチマーク用設定を作成"""
        benchmark_config = {
            "system": {
                "max_workers": psutil.cpu_count(),
                "memory_limit": f"{int(psutil.virtual_memory().total / (1024**3))}GB",
                "gpu_enabled": False,
            },
            "directories": {
                "input": f"{self.temp_dir}/input",
                "output": f"{self.temp_dir}/output",
                "temp": f"{self.temp_dir}/temp",
                "models": f"{self.temp_dir}/models",
            },
        }

        # 設定を更新
        self.config_manager.config.update(benchmark_config)

    def create_test_video(
        self, path: str, duration_seconds: int, fps: int = 30, resolution: tuple = (1920, 1080)
    ) -> bool:
        """テスト用動画を作成"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(path, fourcc, fps, resolution)

            total_frames = duration_seconds * fps
            for i in range(total_frames):
                # 複雑なフレームを生成
                frame = np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)

                # 麻雀牌のような矩形を追加
                for j in range(20):
                    x = np.random.randint(0, resolution[0] - 100)
                    y = np.random.randint(0, resolution[1] - 150)
                    color = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    )
                    cv2.rectangle(frame, (x, y), (x + 80, y + 120), color, -1)

                out.write(frame)

            out.release()
            return True

        except Exception as e:
            self.logger.error(f"Failed to create test video: {e}")
            return False

    def benchmark_video_processing(self) -> BenchmarkResult:
        """動画処理のベンチマーク"""
        test_name = "Video Processing"
        self.logger.info(f"Running benchmark: {test_name}")

        try:
            # テスト動画作成
            video_path = os.path.join(self.temp_dir, "input/benchmark_video.mp4")
            if not self.create_test_video(video_path, duration_seconds=30, resolution=(1280, 720)):
                raise Exception("Failed to create test video")

            # メモリ使用量監視開始
            initial_memory = psutil.Process().memory_info().rss / (1024**2)  # MB

            # CPU使用率監視
            cpu_percent_start = psutil.cpu_percent(interval=None)

            # 処理開始
            start_time = time.time()

            # 動画処理をシミュレート（実際のVideoProcessorの代わり）
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            processed_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # フレーム処理をシミュレート
                processed_frame = cv2.GaussianBlur(frame, (5, 5), 0)
                processed_frames.append(processed_frame)
                frame_count += 1

                # メモリ制限チェック
                current_memory = psutil.Process().memory_info().rss / (1024**2)
                if current_memory - initial_memory > 2000:  # 2GB制限
                    self.logger.warning("Memory limit reached, stopping processing")
                    break

            cap.release()

            # 処理終了
            processing_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / (1024**2)
            cpu_percent_end = psutil.cpu_percent(interval=None)

            # スループット計算
            throughput = frame_count / processing_time if processing_time > 0 else 0

            # メモリ使用量
            memory_usage = final_memory - initial_memory

            # CPU使用率（平均）
            cpu_usage = (cpu_percent_start + cpu_percent_end) / 2

            result = BenchmarkResult(
                test_name=test_name,
                success=True,
                processing_time=processing_time,
                throughput=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                additional_metrics={
                    "frames_processed": frame_count,
                    "fps": throughput,
                    "video_duration": 30,
                    "resolution": "1280x720",
                },
            )

            self.logger.info(
                f"{test_name} completed: {throughput:.2f} FPS, {memory_usage:.1f}MB memory"
            )
            return result

        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                processing_time=0,
                throughput=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e),
            )

    def benchmark_memory_performance(self) -> BenchmarkResult:
        """メモリパフォーマンスのベンチマーク"""
        test_name = "Memory Performance"
        self.logger.info(f"Running benchmark: {test_name}")

        try:
            memory_optimizer = MemoryOptimizer(self.config_manager)

            # 初期メモリ使用量
            initial_memory = memory_optimizer.get_memory_usage()

            start_time = time.time()

            # メモリ集約的な処理をシミュレート
            large_arrays = []
            max_arrays = 50
            arrays_created = 0

            for i in range(max_arrays):
                # 50MB相当のデータを作成
                array = np.random.random((1000, 1000, 5)).astype(np.float32)
                large_arrays.append(array)
                arrays_created += 1

                # メモリ使用量チェック
                current_memory = memory_optimizer.get_memory_usage()
                memory_increase = current_memory - initial_memory

                if memory_increase > 3000:  # 3GB制限
                    self.logger.info(f"Memory limit reached at iteration {i}")
                    break

                # 少し待機
                time.sleep(0.01)

            # メモリクリーンアップテスト
            cleanup_start = time.time()
            memory_optimizer.cleanup_memory()
            del large_arrays
            cleanup_time = time.time() - cleanup_start

            processing_time = time.time() - start_time
            final_memory = memory_optimizer.get_memory_usage()

            # メモリ効率計算
            peak_memory = max(initial_memory, final_memory)
            memory_efficiency = (
                (peak_memory - initial_memory) / arrays_created if arrays_created > 0 else 0
            )

            result = BenchmarkResult(
                test_name=test_name,
                success=True,
                processing_time=processing_time,
                throughput=arrays_created / processing_time if processing_time > 0 else 0,
                memory_usage_mb=peak_memory - initial_memory,
                cpu_usage_percent=psutil.cpu_percent(),
                additional_metrics={
                    "arrays_created": arrays_created,
                    "cleanup_time": cleanup_time,
                    "memory_efficiency_mb_per_array": memory_efficiency,
                    "peak_memory_mb": peak_memory,
                },
            )

            self.logger.info(
                f"{test_name} completed: {arrays_created} arrays, {memory_efficiency:.1f}MB/array"
            )
            return result

        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                processing_time=0,
                throughput=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e),
            )

    def benchmark_cpu_performance(self) -> BenchmarkResult:
        """CPU パフォーマンスのベンチマーク"""
        test_name = "CPU Performance"
        self.logger.info(f"Running benchmark: {test_name}")

        try:
            start_time = time.time()

            # CPU集約的な処理
            def cpu_intensive_task(iterations: int) -> int:
                result = 0
                for i in range(iterations):
                    result += sum(range(1000))
                return result

            # シングルスレッド性能
            single_start = time.time()
            single_result = cpu_intensive_task(10000)
            single_time = time.time() - single_start

            # マルチスレッド性能
            multi_start = time.time()
            threads = []
            thread_count = psutil.cpu_count()

            for i in range(thread_count):
                thread = threading.Thread(target=cpu_intensive_task, args=(10000 // thread_count,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            multi_time = time.time() - multi_start

            processing_time = time.time() - start_time

            # スピードアップ計算
            speedup = single_time / multi_time if multi_time > 0 else 0
            efficiency = speedup / thread_count if thread_count > 0 else 0

            result = BenchmarkResult(
                test_name=test_name,
                success=True,
                processing_time=processing_time,
                throughput=1 / processing_time if processing_time > 0 else 0,
                memory_usage_mb=psutil.Process().memory_info().rss / (1024**2),
                cpu_usage_percent=psutil.cpu_percent(),
                additional_metrics={
                    "single_thread_time": single_time,
                    "multi_thread_time": multi_time,
                    "thread_count": thread_count,
                    "speedup": speedup,
                    "efficiency": efficiency,
                },
            )

            self.logger.info(
                f"{test_name} completed: {speedup:.2f}x speedup, {efficiency:.2%} efficiency"
            )
            return result

        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                processing_time=0,
                throughput=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e),
            )

    def benchmark_io_performance(self) -> BenchmarkResult:
        """I/O パフォーマンスのベンチマーク"""
        test_name = "I/O Performance"
        self.logger.info(f"Running benchmark: {test_name}")

        try:
            start_time = time.time()

            # ファイル書き込み性能
            write_start = time.time()
            test_file = os.path.join(self.temp_dir, "temp/io_test.dat")

            # 100MB のデータを書き込み
            data_size_mb = 100
            chunk_size = 1024 * 1024  # 1MB chunks

            with open(test_file, "wb") as f:
                for i in range(data_size_mb):
                    data = np.random.bytes(chunk_size)
                    f.write(data)

            write_time = time.time() - write_start

            # ファイル読み込み性能
            read_start = time.time()

            with open(test_file, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

            read_time = time.time() - read_start

            # ファイル削除
            os.remove(test_file)

            processing_time = time.time() - start_time

            # スループット計算 (MB/s)
            write_throughput = data_size_mb / write_time if write_time > 0 else 0
            read_throughput = data_size_mb / read_time if read_time > 0 else 0

            result = BenchmarkResult(
                test_name=test_name,
                success=True,
                processing_time=processing_time,
                throughput=(write_throughput + read_throughput) / 2,
                memory_usage_mb=psutil.Process().memory_info().rss / (1024**2),
                cpu_usage_percent=psutil.cpu_percent(),
                additional_metrics={
                    "data_size_mb": data_size_mb,
                    "write_time": write_time,
                    "read_time": read_time,
                    "write_throughput_mbps": write_throughput,
                    "read_throughput_mbps": read_throughput,
                },
            )

            self.logger.info(
                f"{test_name} completed: Write {write_throughput:.1f}MB/s, Read {read_throughput:.1f}MB/s"
            )
            return result

        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                processing_time=0,
                throughput=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e),
            )

    def benchmark_system_optimization(self) -> BenchmarkResult:
        """システム最適化のベンチマーク"""
        test_name = "System Optimization"
        self.logger.info(f"Running benchmark: {test_name}")

        try:
            performance_optimizer = PerformanceOptimizer(self.config_manager)

            start_time = time.time()

            # 最適化前のメトリクス
            before_metrics = performance_optimizer.get_current_metrics()

            # システム最適化実行
            optimization_result = performance_optimizer.optimize_system()

            # 最適化後のメトリクス
            after_metrics = performance_optimizer.get_current_metrics()

            processing_time = time.time() - start_time

            # 改善度計算
            cpu_improvement = (
                before_metrics.cpu_usage - after_metrics.cpu_usage
                if hasattr(before_metrics, "cpu_usage") and hasattr(after_metrics, "cpu_usage")
                else 0
            )
            memory_improvement = (
                before_metrics.memory_usage - after_metrics.memory_usage
                if hasattr(before_metrics, "memory_usage")
                and hasattr(after_metrics, "memory_usage")
                else 0
            )

            result = BenchmarkResult(
                test_name=test_name,
                success=optimization_result.success
                if hasattr(optimization_result, "success")
                else True,
                processing_time=processing_time,
                throughput=1 / processing_time if processing_time > 0 else 0,
                memory_usage_mb=after_metrics.memory_usage
                if hasattr(after_metrics, "memory_usage")
                else 0,
                cpu_usage_percent=after_metrics.cpu_usage
                if hasattr(after_metrics, "cpu_usage")
                else 0,
                additional_metrics={
                    "cpu_improvement": cpu_improvement,
                    "memory_improvement": memory_improvement,
                    "optimization_type": getattr(
                        optimization_result, "optimization_type", "general"
                    ),
                    "recommendations_count": len(
                        performance_optimizer.get_optimization_recommendations()
                    ),
                },
            )

            self.logger.info(
                f"{test_name} completed: CPU {cpu_improvement:+.1f}%, Memory {memory_improvement:+.1f}%"
            )
            return result

        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            return BenchmarkResult(
                test_name=test_name,
                success=False,
                processing_time=0,
                throughput=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e),
            )

    def run_all_benchmarks(self) -> dict[str, Any]:
        """全ベンチマークを実行"""
        self.logger.info("Starting comprehensive system benchmark")
        self.logger.info("=" * 60)

        if not self.setup_benchmark_environment():
            return {"success": False, "error": "Failed to setup benchmark environment"}

        # ベンチマーク実行
        benchmarks = [
            self.benchmark_video_processing,
            self.benchmark_memory_performance,
            self.benchmark_cpu_performance,
            self.benchmark_io_performance,
            self.benchmark_system_optimization,
        ]

        for benchmark_func in benchmarks:
            try:
                result = benchmark_func()
                self.results.append(result)

                if result.success:
                    self.logger.info(f"✅ {result.test_name}: {result.processing_time:.2f}s")
                else:
                    self.logger.error(f"❌ {result.test_name}: {result.error_message}")

            except Exception as e:
                self.logger.error(f"Benchmark {benchmark_func.__name__} failed: {e}")

        # 結果サマリー生成
        summary = self._generate_benchmark_summary()

        # レポート保存
        report_path = self._save_benchmark_report(summary)

        self.logger.info("=" * 60)
        self.logger.info("Benchmark completed")
        self.logger.info(f"Report saved: {report_path}")

        return {
            "success": True,
            "results": [result.__dict__ for result in self.results],
            "summary": summary,
            "report_path": report_path,
            "system_info": self.system_info,
        }

    def _generate_benchmark_summary(self) -> dict[str, Any]:
        """ベンチマーク結果サマリーを生成"""
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]

        summary = {
            "total_tests": len(self.results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.results) if self.results else 0,
            "total_time": sum(r.processing_time for r in self.results),
            "average_throughput": np.mean([r.throughput for r in successful_tests])
            if successful_tests
            else 0,
            "average_memory_usage": np.mean([r.memory_usage_mb for r in successful_tests])
            if successful_tests
            else 0,
            "average_cpu_usage": np.mean([r.cpu_usage_percent for r in successful_tests])
            if successful_tests
            else 0,
            "performance_score": self._calculate_performance_score(),
            "recommendations": self._generate_recommendations(),
        }

        return summary

    def _calculate_performance_score(self) -> float:
        """パフォーマンススコアを計算"""
        if not self.results:
            return 0.0

        # 各テストの重み
        weights = {
            "Video Processing": 0.3,
            "Memory Performance": 0.2,
            "CPU Performance": 0.2,
            "I/O Performance": 0.2,
            "System Optimization": 0.1,
        }

        total_score = 0.0
        total_weight = 0.0

        for result in self.results:
            if result.success and result.test_name in weights:
                # 正規化されたスコア（0-100）
                normalized_score = min(100, result.throughput * 10)  # 簡易的な正規化
                weighted_score = normalized_score * weights[result.test_name]
                total_score += weighted_score
                total_weight += weights[result.test_name]

        return total_score / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(self) -> list[str]:
        """推奨事項を生成"""
        recommendations = []

        for result in self.results:
            if not result.success:
                recommendations.append(f"Fix issues in {result.test_name}: {result.error_message}")
            elif result.throughput < 1.0:
                recommendations.append(
                    f"Consider optimizing {result.test_name} for better throughput"
                )
            elif result.memory_usage_mb > 1000:
                recommendations.append(f"Optimize memory usage in {result.test_name}")

        # システム全体の推奨事項
        avg_cpu = np.mean([r.cpu_usage_percent for r in self.results if r.success])
        if avg_cpu > 80:
            recommendations.append("Consider reducing CPU load or increasing CPU resources")

        avg_memory = np.mean([r.memory_usage_mb for r in self.results if r.success])
        if avg_memory > 2000:
            recommendations.append("Consider optimizing memory usage or increasing RAM")

        return recommendations

    def _save_benchmark_report(self, summary: dict[str, Any]) -> str:
        """ベンチマークレポートを保存"""
        report_data = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "system_info": self.system_info,
            },
            "summary": summary,
            "detailed_results": [result.__dict__ for result in self.results],
        }

        report_path = os.path.join(self.temp_dir, "benchmark_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return report_path


def main():
    """メイン関数"""
    print("麻雀牌譜作成システム - ベンチマークテスト")
    print("=" * 50)

    try:
        # ベンチマーク実行
        benchmark = SystemBenchmark()
        result = benchmark.run_all_benchmarks()

        if result["success"]:
            print("\n✅ ベンチマークが正常に完了しました！")
            print(f"📊 総テスト数: {result['summary']['total_tests']}")
            print(f"✅ 成功: {result['summary']['successful_tests']}")
            print(f"❌ 失敗: {result['summary']['failed_tests']}")
            print(f"📈 成功率: {result['summary']['success_rate']:.2%}")
            print(f"⏱️  総時間: {result['summary']['total_time']:.2f}秒")
            print(f"🏆 パフォーマンススコア: {result['summary']['performance_score']:.1f}")

            # システム情報
            sys_info = result["system_info"]
            print("\n💻 システム情報:")
            print(
                f"  - CPU: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)"
            )
            print(f"  - メモリ: {sys_info['memory_total_gb']:.1f}GB")
            print(f"  - プラットフォーム: {sys_info['platform']}")

            # 推奨事項
            if result["summary"]["recommendations"]:
                print("\n💡 推奨事項:")
                for rec in result["summary"]["recommendations"]:
                    print(f"  - {rec}")

            print(f"\n📄 詳細レポート: {result['report_path']}")
        else:
            print(f"\n❌ ベンチマークが失敗しました: {result.get('error', 'Unknown error')}")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\n⏹️  ベンチマークが中断されました")
        return 130
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
