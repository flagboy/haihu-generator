"""
シーン分析ユーティリティ

対局シーンの詳細分析とレポート生成
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ....utils.logger import LoggerMixin
from ..core.scene_detector import SceneDetector


class SceneAnalyzer(LoggerMixin):
    """シーン分析クラス"""

    def __init__(self, detector: SceneDetector | None = None):
        """
        初期化

        Args:
            detector: シーン検出器
        """
        self.detector = detector or SceneDetector()
        self.logger.info("SceneAnalyzer初期化完了")

    def analyze_video(
        self, video_path: str, output_dir: str = "analysis/scene_analysis"
    ) -> dict[str, any]:
        """
        動画の対局シーンを分析

        Args:
            video_path: 動画ファイルパス
            output_dir: 分析結果の出力ディレクトリ

        Returns:
            分析結果
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # シーン検出
        self.logger.info(f"動画分析開始: {video_path}")
        segments = self.detector.detect_scenes(video_path)

        # 動画情報を取得
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # 統計を計算
        stats = self._calculate_statistics(segments, total_frames, fps)

        # 可視化
        self._visualize_timeline(segments, total_frames, output_dir / "timeline.png")

        self._visualize_distribution(segments, output_dir / "distribution.png")

        # レポート生成
        report = {
            "video_info": {
                "path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
            },
            "statistics": stats,
            "segments": [
                {
                    "type": seg.scene_type,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "start_time": seg.start_frame / fps,
                    "end_time": seg.end_frame / fps,
                    "duration": seg.duration_frames / fps,
                    "confidence": seg.confidence,
                }
                for seg in segments
            ],
        }

        # レポートを保存
        report_path = output_dir / "analysis_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"分析完了: 結果を {output_dir} に保存")

        return report

    def _calculate_statistics(
        self, segments: list, total_frames: int, fps: float
    ) -> dict[str, any]:
        """統計を計算"""
        game_segments = [s for s in segments if s.scene_type == "game"]
        non_game_segments = [s for s in segments if s.scene_type == "non_game"]

        game_frames = sum(s.duration_frames for s in game_segments)
        non_game_frames = sum(s.duration_frames for s in non_game_segments)

        # セグメント長の統計
        game_durations = [s.duration_frames / fps for s in game_segments]
        non_game_durations = [s.duration_frames / fps for s in non_game_segments]

        stats = {
            "total_segments": len(segments),
            "game_segments": len(game_segments),
            "non_game_segments": len(non_game_segments),
            "game_frames": game_frames,
            "non_game_frames": non_game_frames,
            "unclassified_frames": total_frames - game_frames - non_game_frames,
            "game_ratio": game_frames / total_frames if total_frames > 0 else 0,
            "average_game_duration": np.mean(game_durations) if game_durations else 0,
            "average_non_game_duration": np.mean(non_game_durations) if non_game_durations else 0,
            "longest_game_segment": max(game_durations) if game_durations else 0,
            "longest_non_game_segment": max(non_game_durations) if non_game_durations else 0,
            "efficiency_gain": 1.0 - (game_frames / total_frames) if total_frames > 0 else 0,
        }

        return stats

    def _visualize_timeline(self, segments: list, total_frames: int, output_path: Path):
        """タイムラインを可視化"""
        fig, ax = plt.subplots(figsize=(14, 4))

        # セグメントを描画
        for segment in segments:
            color = "green" if segment.scene_type == "game" else "red"
            alpha = segment.confidence if segment.confidence else 0.8

            ax.barh(
                0,
                segment.duration_frames,
                left=segment.start_frame,
                height=1,
                color=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xlim(0, total_frames)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("フレーム番号")
        ax.set_title("対局シーンタイムライン（緑: 対局, 赤: 非対局）")
        ax.set_yticks([])

        # グリッド
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _visualize_distribution(self, segments: list, output_path: Path):
        """セグメント長の分布を可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # セグメント長を収集
        game_lengths = [s.duration_frames for s in segments if s.scene_type == "game"]
        non_game_lengths = [s.duration_frames for s in segments if s.scene_type == "non_game"]

        # ヒストグラム
        if game_lengths:
            ax1.hist(game_lengths, bins=20, color="green", alpha=0.7, label="対局")
        if non_game_lengths:
            ax1.hist(non_game_lengths, bins=20, color="red", alpha=0.7, label="非対局")

        ax1.set_xlabel("セグメント長（フレーム）")
        ax1.set_ylabel("頻度")
        ax1.set_title("セグメント長の分布")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 円グラフ
        total_game = sum(game_lengths) if game_lengths else 0
        total_non_game = sum(non_game_lengths) if non_game_lengths else 0

        if total_game > 0 or total_non_game > 0:
            ax2.pie(
                [total_game, total_non_game],
                labels=["対局", "非対局"],
                colors=["green", "red"],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax2.set_title("フレーム数の割合")
        else:
            ax2.text(0.5, 0.5, "データなし", ha="center", va="center")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def compare_videos(
        self, video_paths: list[str], output_dir: str = "analysis/comparison"
    ) -> dict[str, any]:
        """
        複数動画の対局シーンを比較

        Args:
            video_paths: 動画ファイルパスのリスト
            output_dir: 分析結果の出力ディレクトリ

        Returns:
            比較結果
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        # 各動画を分析
        for video_path in video_paths:
            video_name = Path(video_path).name
            video_output_dir = output_dir / video_name.replace(".", "_")

            result = self.analyze_video(video_path, str(video_output_dir))
            result["video_name"] = video_name
            results.append(result)

        # 比較レポート生成
        comparison = self._generate_comparison_report(results)

        # 比較可視化
        self._visualize_comparison(results, output_dir / "comparison.png")

        # レポート保存
        comparison_path = output_dir / "comparison_report.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        return comparison

    def _generate_comparison_report(self, results: list[dict]) -> dict[str, any]:
        """比較レポート生成"""
        comparison = {"video_count": len(results), "videos": []}

        for result in results:
            stats = result["statistics"]
            comparison["videos"].append(
                {
                    "name": result["video_name"],
                    "duration": result["video_info"]["duration"],
                    "game_ratio": stats["game_ratio"],
                    "efficiency_gain": stats["efficiency_gain"],
                    "game_segments": stats["game_segments"],
                    "average_game_duration": stats["average_game_duration"],
                }
            )

        # 全体統計
        game_ratios = [r["statistics"]["game_ratio"] for r in results]
        efficiency_gains = [r["statistics"]["efficiency_gain"] for r in results]

        comparison["overall"] = {
            "average_game_ratio": np.mean(game_ratios),
            "std_game_ratio": np.std(game_ratios),
            "average_efficiency_gain": np.mean(efficiency_gains),
            "best_efficiency": max(efficiency_gains),
            "worst_efficiency": min(efficiency_gains),
        }

        return comparison

    def _visualize_comparison(self, results: list[dict], output_path: Path):
        """比較結果を可視化"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        video_names = [
            r["video_name"][:20] + "..." if len(r["video_name"]) > 20 else r["video_name"]
            for r in results
        ]
        game_ratios = [r["statistics"]["game_ratio"] for r in results]
        efficiency_gains = [r["statistics"]["efficiency_gain"] for r in results]

        # 対局率の比較
        x = np.arange(len(video_names))
        ax1.bar(x, game_ratios, color="green", alpha=0.7)
        ax1.set_xlabel("動画")
        ax1.set_ylabel("対局画面の割合")
        ax1.set_title("動画別対局画面率")
        ax1.set_xticks(x)
        ax1.set_xticklabels(video_names, rotation=45, ha="right")
        ax1.grid(True, axis="y", alpha=0.3)

        # 平均線
        avg_ratio = np.mean(game_ratios)
        ax1.axhline(y=avg_ratio, color="red", linestyle="--", label=f"平均: {avg_ratio:.2%}")
        ax1.legend()

        # 効率化の比較
        ax2.bar(x, efficiency_gains, color="blue", alpha=0.7)
        ax2.set_xlabel("動画")
        ax2.set_ylabel("処理削減率")
        ax2.set_title("フレームスキップによる効率化")
        ax2.set_xticks(x)
        ax2.set_xticklabels(video_names, rotation=45, ha="right")
        ax2.grid(True, axis="y", alpha=0.3)

        # 値をバーの上に表示
        for i, v in enumerate(efficiency_gains):
            ax2.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
