#!/usr/bin/env python3
"""
統合テスト実行スクリプト
システム全体の統合テストを実行し、結果をレポート
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import get_logger


class IntegrationTestRunner:
    """統合テスト実行クラス"""

    def __init__(self):
        """初期化"""
        self.logger = get_logger(__name__)
        self.test_results = []
        self.start_time = None
        self.end_time = None

    def run_pytest_tests(self) -> dict[str, Any]:
        """pytestテストを実行"""
        self.logger.info("Running pytest integration tests...")

        try:
            # 統合テストを実行
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/",
                "-v",
                "--tb=short",
                "--cov=src",
                "--cov-report=json",
                "--cov-report=html",
                "--junit-xml=test_results.xml",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分でタイムアウト
            )

            # カバレッジ結果を読み込み
            coverage_data = {}
            if os.path.exists("coverage.json"):
                with open("coverage.json") as f:
                    coverage_data = json.load(f)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                "test_type": "pytest_integration",
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timed out",
                "test_type": "pytest_integration",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "test_type": "pytest_integration"}

    def run_performance_tests(self) -> dict[str, Any]:
        """パフォーマンステストを実行"""
        self.logger.info("Running performance tests...")

        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/test_performance.py",
                "-v",
                "--tb=short",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15分でタイムアウト
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "test_type": "performance",
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Performance test timed out",
                "test_type": "performance",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "test_type": "performance"}

    def run_end_to_end_tests(self) -> dict[str, Any]:
        """エンドツーエンドテストを実行"""
        self.logger.info("Running end-to-end tests...")

        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/test_end_to_end.py",
                "-v",
                "--tb=short",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,  # 20分でタイムアウト
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "test_type": "end_to_end",
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "End-to-end test timed out",
                "test_type": "end_to_end",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "test_type": "end_to_end"}

    def run_demo_workflow(self) -> dict[str, Any]:
        """デモワークフローを実行"""
        self.logger.info("Running demo workflow...")

        try:
            cmd = [sys.executable, "demo_complete_workflow.py"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分でタイムアウト
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "test_type": "demo_workflow",
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Demo workflow timed out",
                "test_type": "demo_workflow",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "test_type": "demo_workflow"}

    def run_benchmark_tests(self) -> dict[str, Any]:
        """ベンチマークテストを実行"""
        self.logger.info("Running benchmark tests...")

        try:
            cmd = [sys.executable, "benchmark_system.py"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15分でタイムアウト
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "test_type": "benchmark",
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Benchmark test timed out", "test_type": "benchmark"}
        except Exception as e:
            return {"success": False, "error": str(e), "test_type": "benchmark"}

    def run_docker_tests(self) -> dict[str, Any]:
        """Dockerテストを実行"""
        self.logger.info("Running Docker tests...")

        try:
            # Dockerイメージをビルド
            build_cmd = ["docker", "build", "-t", "mahjong-system-test", "."]
            build_result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=600)

            if build_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Docker build failed: {build_result.stderr}",
                    "test_type": "docker",
                }

            # Dockerコンテナでテストを実行
            test_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{os.getcwd()}:/app",
                "mahjong-system-test",
                "python",
                "main.py",
                "status",
            ]

            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=300)

            return {
                "success": test_result.returncode == 0,
                "stdout": test_result.stdout,
                "stderr": test_result.stderr,
                "return_code": test_result.returncode,
                "test_type": "docker",
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Docker test timed out", "test_type": "docker"}
        except Exception as e:
            return {"success": False, "error": str(e), "test_type": "docker"}

    def run_code_quality_checks(self) -> dict[str, Any]:
        """コード品質チェックを実行"""
        self.logger.info("Running code quality checks...")

        results = {}

        # Flake8
        try:
            flake8_result = subprocess.run(
                [sys.executable, "-m", "flake8", "src", "tests"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            results["flake8"] = {
                "success": flake8_result.returncode == 0,
                "output": flake8_result.stdout + flake8_result.stderr,
            }
        except Exception as e:
            results["flake8"] = {"success": False, "error": str(e)}

        # Black
        try:
            black_result = subprocess.run(
                [sys.executable, "-m", "black", "--check", "src", "tests"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            results["black"] = {
                "success": black_result.returncode == 0,
                "output": black_result.stdout + black_result.stderr,
            }
        except Exception as e:
            results["black"] = {"success": False, "error": str(e)}

        # isort
        try:
            isort_result = subprocess.run(
                [sys.executable, "-m", "isort", "--check-only", "src", "tests"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            results["isort"] = {
                "success": isort_result.returncode == 0,
                "output": isort_result.stdout + isort_result.stderr,
            }
        except Exception as e:
            results["isort"] = {"success": False, "error": str(e)}

        # 全体の成功判定
        overall_success = all(result.get("success", False) for result in results.values())

        return {"success": overall_success, "results": results, "test_type": "code_quality"}

    def run_all_tests(self) -> dict[str, Any]:
        """全テストを実行"""
        self.start_time = time.time()
        self.logger.info("Starting comprehensive integration test suite")
        self.logger.info("=" * 60)

        # テスト実行順序
        test_functions = [
            ("Code Quality Checks", self.run_code_quality_checks),
            ("Unit & Integration Tests", self.run_pytest_tests),
            ("Performance Tests", self.run_performance_tests),
            ("End-to-End Tests", self.run_end_to_end_tests),
            ("Demo Workflow", self.run_demo_workflow),
            ("Benchmark Tests", self.run_benchmark_tests),
            ("Docker Tests", self.run_docker_tests),
        ]

        for test_name, test_func in test_functions:
            self.logger.info(f"Running {test_name}...")

            test_start = time.time()
            result = test_func()
            test_duration = time.time() - test_start

            result["test_name"] = test_name
            result["duration"] = test_duration
            result["timestamp"] = datetime.now().isoformat()

            self.test_results.append(result)

            if result["success"]:
                self.logger.info(f"✅ {test_name} passed ({test_duration:.2f}s)")
            else:
                self.logger.error(f"❌ {test_name} failed ({test_duration:.2f}s)")
                if "error" in result:
                    self.logger.error(f"   Error: {result['error']}")

        self.end_time = time.time()

        # 結果サマリー生成
        summary = self._generate_summary()

        # レポート保存
        report_path = self._save_test_report(summary)

        self.logger.info("=" * 60)
        self.logger.info("Integration test suite completed")
        self.logger.info(f"Total time: {summary['total_duration']:.2f}s")
        self.logger.info(f"Tests passed: {summary['passed_tests']}/{summary['total_tests']}")
        self.logger.info(f"Success rate: {summary['success_rate']:.2%}")
        self.logger.info(f"Report saved: {report_path}")

        return {
            "success": summary["success_rate"] >= 0.8,  # 80%以上で成功
            "summary": summary,
            "results": self.test_results,
            "report_path": report_path,
        }

    def _generate_summary(self) -> dict[str, Any]:
        """テスト結果サマリーを生成"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests

        total_duration = self.end_time - self.start_time if self.start_time and self.end_time else 0

        # カバレッジ情報
        coverage = 0
        for result in self.test_results:
            if result.get("test_type") == "pytest_integration" and "coverage" in result:
                coverage = result["coverage"]
                break

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration,
            "coverage_percentage": coverage,
            "timestamp": datetime.now().isoformat(),
            "failed_test_names": [
                result["test_name"] for result in self.test_results if not result["success"]
            ],
        }

    def _save_test_report(self, summary: dict[str, Any]) -> str:
        """テストレポートを保存"""
        report_data = {
            "summary": summary,
            "detailed_results": self.test_results,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": os.getcwd(),
            },
        }

        # レポートディレクトリ作成
        report_dir = Path("test_reports")
        report_dir.mkdir(exist_ok=True)

        # レポートファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"integration_test_report_{timestamp}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return str(report_path)


def main():
    """メイン関数"""
    print("麻雀牌譜作成システム - 統合テスト実行")
    print("=" * 50)

    try:
        # 統合テスト実行
        runner = IntegrationTestRunner()
        result = runner.run_all_tests()

        if result["success"]:
            print("\n🎉 統合テストが正常に完了しました！")
            print(f"📊 成功率: {result['summary']['success_rate']:.2%}")
            print(f"⏱️  総時間: {result['summary']['total_duration']:.2f}秒")
            print(f"📈 カバレッジ: {result['summary']['coverage_percentage']:.1f}%")

            if result["summary"]["failed_tests"] > 0:
                print(f"\n⚠️  失敗したテスト ({result['summary']['failed_tests']}件):")
                for test_name in result["summary"]["failed_test_names"]:
                    print(f"  - {test_name}")

            print(f"\n📄 詳細レポート: {result['report_path']}")
        else:
            print("\n❌ 統合テストが失敗しました")
            print(f"📊 成功率: {result['summary']['success_rate']:.2%}")
            print(f"❌ 失敗: {result['summary']['failed_tests']}件")

            print("\n失敗したテスト:")
            for test_name in result["summary"]["failed_test_names"]:
                print(f"  - {test_name}")

            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\n⏹️  テストが中断されました")
        return 130
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
