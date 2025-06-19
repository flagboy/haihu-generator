"""
統合モデル訓練システム

データ拡張、ラベリング、YOLOv8を統合した訓練システム
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

# プロジェクトのルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.detection.yolov8_detector import YOLOv8TileDetector  # noqa: E402
from src.training.augmentation.unified_augmentor import UnifiedAugmentor  # noqa: E402
from src.training.labeling.batch_labeler import BatchLabeler  # noqa: E402


class UnifiedModelTrainer:
    """データ拡張、ラベリング、YOLOv8を統合した訓練システム"""

    def __init__(self, config_path: str | None = None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.augmentor = UnifiedAugmentor(self.config.get("augmentation", {}))
        self.detector = YOLOv8TileDetector()
        self.batch_labeler = BatchLabeler(None)

        # 作業ディレクトリの設定
        self.work_dir = Path(self.config.get("work_dir", "work/training"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """設定ファイルを読み込む"""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)

        # デフォルト設定
        return {
            "augmentation": {
                "augmentation_factor": 20,
                "enable_red_dora": True,
                "output_format": "yolo",
            },
            "training": {"batch_size": 16, "epochs": 100, "imgsz": 640, "device": "auto"},
            "target_accuracy": 0.5,
        }

    def create_initial_model(
        self, raw_data_path: str, output_model_path: str, target_accuracy: float = 0.5
    ) -> dict[str, Any]:
        """
        初期モデル作成の完全なワークフロー

        Args:
            raw_data_path: 生データのパス
            output_model_path: 出力モデルのパス
            target_accuracy: 目標精度

        Returns:
            訓練結果の統計
        """
        print("=" * 60)
        print("初期モデル作成ワークフローを開始")
        print("=" * 60)

        start_time = datetime.now()

        # Step 1: データ拡張
        print("\n[Step 1/4] データ拡張を実行中...")
        augmented_path = self.work_dir / "augmented_data"
        augmentation_report = self.augmentor.augment_dataset(raw_data_path, str(augmented_path))

        # Step 2: YOLOv8形式への変換
        print("\n[Step 2/4] YOLOv8形式に変換中...")
        yolo_dataset_path = self._prepare_yolo_dataset(augmented_path)

        # Step 3: 初期訓練
        print("\n[Step 3/4] 初期モデルの訓練を開始...")
        training_results = self._train_initial_model(yolo_dataset_path, output_model_path)

        # Step 4: 評価とレポート生成
        print("\n[Step 4/4] モデルの評価中...")
        metrics = self._evaluate_model(output_model_path, yolo_dataset_path)

        # 処理時間
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # 結果の表示
        print("\n" + "=" * 60)
        print("訓練完了！")
        print("=" * 60)
        print(f"処理時間: {processing_time:.1f}秒")
        print(f"mAP@0.5: {metrics['mAP50']:.3f}")
        print(f"mAP@0.5:0.95: {metrics['mAP']:.3f}")

        if metrics["mAP"] >= target_accuracy:
            print(f"✅ 目標精度 {target_accuracy} を達成しました！")
        else:
            print(f"⚠️ 目標精度未達: {metrics['mAP']:.3f} < {target_accuracy}")

        # 総合レポートの生成
        report = self._generate_comprehensive_report(
            augmentation_report, training_results, metrics, output_model_path, processing_time
        )

        return report

    def _prepare_yolo_dataset(self, augmented_path: Path) -> str:
        """拡張データからYOLOv8データセットを準備"""
        yolo_path = self.work_dir / "yolo_dataset"

        # 既存のディレクトリがあれば削除
        if yolo_path.exists():
            shutil.rmtree(yolo_path)

        # YOLOv8形式に変換
        dataset_yaml = self.detector.prepare_training_data(str(augmented_path), str(yolo_path))

        return dataset_yaml

    def _train_initial_model(self, data_yaml: str, output_path: str) -> dict[str, Any]:
        """初期モデルの訓練"""
        # 訓練パラメータ
        train_config = self.config.get("training", {})

        # 初期モデル用の調整
        initial_config = {
            "epochs": min(train_config.get("epochs", 100), 50),  # 初期は少なめ
            "batch": train_config.get("batch_size", 16),
            "imgsz": train_config.get("imgsz", 640),
            "patience": 20,  # 早期停止を早めに
            "project": str(self.work_dir / "runs"),
            "name": "initial_model",
        }

        # 訓練実行
        _ = self.detector.train(data_yaml, **initial_config)

        # 最良モデルをコピー
        best_model = (
            Path(initial_config["project"]) / initial_config["name"] / "weights" / "best.pt"
        )
        if best_model.exists():
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_model, output_path)
            print(f"最良モデルを保存: {output_path}")

        return {
            "epochs_trained": initial_config["epochs"],
            "best_model_path": str(best_model),
            "output_model_path": output_path,
        }

    def _evaluate_model(self, model_path: str, data_yaml: str) -> dict[str, float]:
        """モデルの評価"""
        # モデルを読み込み
        self.detector = YOLOv8TileDetector(model_path)

        # 評価実行
        metrics = self.detector.evaluate(data_yaml)

        # クラスごとの評価の取得は将来の拡張のために予約

        return metrics

    def _generate_comprehensive_report(
        self,
        augmentation_report: dict,
        training_results: dict,
        metrics: dict,
        model_path: str,
        processing_time: float,
    ) -> dict[str, Any]:
        """包括的なレポートを生成"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "model_path": model_path,
            "data_augmentation": {
                "original_images": augmentation_report["statistics"]["total_original_images"],
                "augmented_images": augmentation_report["statistics"]["total_augmented_images"],
                "augmentation_factor": augmentation_report["configuration"]["augmentation_factor"],
                "red_dora_generated": augmentation_report["statistics"]["red_dora_generated"],
            },
            "training": {
                "epochs": training_results["epochs_trained"],
                "batch_size": self.config["training"].get("batch_size", 16),
                "image_size": self.config["training"].get("imgsz", 640),
                "device": self.detector.device,
            },
            "evaluation": metrics,
            "recommendations": self._generate_recommendations(metrics),
        }

        # レポートを保存
        report_path = self.work_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Markdownレポートも生成
        self._generate_markdown_report(report)

        return report

    def _generate_recommendations(self, metrics: dict[str, float]) -> list[str]:
        """メトリクスに基づく推奨事項を生成"""
        recommendations = []

        mAP = metrics.get("mAP", 0)
        mAP50 = metrics.get("mAP50", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)

        # 全体的な精度
        if mAP < 0.3:
            recommendations.append("より多くの訓練データが必要です")
            recommendations.append("データ拡張の設定を見直してください")
        elif mAP < 0.5:
            recommendations.append("半自動ラベリングを開始できます")
            recommendations.append("難しいサンプルを追加してください")
        else:
            recommendations.append("実用レベルに近づいています")
            recommendations.append("実環境でのテストを開始してください")

        # Precision vs Recall
        if precision < recall * 0.8:
            recommendations.append("誤検出が多い可能性があります - 信頼度閾値を上げてください")
        elif recall < precision * 0.8:
            recommendations.append(
                "見逃しが多い可能性があります - より多様なデータを追加してください"
            )

        # mAP50 vs mAP
        if mAP50 > mAP * 2:
            recommendations.append("バウンディングボックスの精度を改善する必要があります")

        return recommendations

    def _generate_markdown_report(self, report: dict):
        """Markdownフォーマットのレポートを生成"""
        md_content = f"""# 初期モデル訓練レポート

## 概要
- **生成日時**: {report["timestamp"]}
- **処理時間**: {report["processing_time_seconds"]:.1f}秒
- **モデルパス**: `{report["model_path"]}`

## データ拡張
| 項目 | 値 |
|------|-----|
| 元画像数 | {report["data_augmentation"]["original_images"]} |
| 拡張後画像数 | {report["data_augmentation"]["augmented_images"]} |
| 拡張倍率 | {report["data_augmentation"]["augmentation_factor"]}倍 |
| 赤ドラ生成数 | {report["data_augmentation"]["red_dora_generated"]} |

## 訓練設定
| 項目 | 値 |
|------|-----|
| エポック数 | {report["training"]["epochs"]} |
| バッチサイズ | {report["training"]["batch_size"]} |
| 画像サイズ | {report["training"]["image_size"]} |
| デバイス | {report["training"]["device"]} |

## 評価結果
| メトリクス | 値 |
|------------|-----|
| mAP@0.5 | {report["evaluation"]["mAP50"]:.3f} |
| mAP@0.5:0.95 | {report["evaluation"]["mAP"]:.3f} |
| Precision | {report["evaluation"]["precision"]:.3f} |
| Recall | {report["evaluation"]["recall"]:.3f} |

## 推奨事項
"""

        for recommendation in report["recommendations"]:
            md_content += f"- {recommendation}\n"

        md_content += """
## 次のステップ

1. **データ収集**: より多様な対局動画から追加データを収集
2. **半自動ラベリング**: 現在のモデルを使用して効率的にラベリング
3. **ファインチューニング**: 特定の環境に合わせてモデルを調整
4. **実環境テスト**: 実際の対局動画での性能評価

---
*このレポートは自動生成されました*
"""

        md_path = self.work_dir / "training_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    def fine_tune_model(
        self, model_path: str, additional_data_path: str, output_path: str, epochs: int = 30
    ) -> dict[str, Any]:
        """
        既存モデルのファインチューニング

        Args:
            model_path: 既存モデルのパス
            additional_data_path: 追加データのパス
            output_path: 出力モデルのパス
            epochs: エポック数

        Returns:
            ファインチューニング結果
        """
        print("ファインチューニングを開始...")

        # 既存モデルを読み込み
        self.detector = YOLOv8TileDetector(model_path)

        # 追加データの準備
        yolo_dataset = self._prepare_yolo_dataset(Path(additional_data_path))

        # ファインチューニング実行
        _ = self.detector.train(
            yolo_dataset,
            epochs=epochs,
            project=str(self.work_dir / "runs"),
            name="fine_tuned",
            resume=True,  # 既存の重みから再開
        )

        # モデルを保存
        best_model = Path(self.work_dir) / "runs" / "fine_tuned" / "weights" / "best.pt"
        if best_model.exists():
            shutil.copy(best_model, output_path)

        return {"original_model": model_path, "fine_tuned_model": output_path, "epochs": epochs}

    def batch_inference(
        self, model_path: str, video_path: str, output_dir: str, sample_rate: int = 30
    ) -> dict[str, Any]:
        """
        動画に対するバッチ推論

        Args:
            model_path: モデルパス
            video_path: 動画パス
            output_dir: 出力ディレクトリ
            sample_rate: サンプリングレート（フレーム）

        Returns:
            推論結果の統計
        """
        # モデルを読み込み
        self.detector = YOLOv8TileDetector(model_path)

        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 動画を開く
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        detection_count = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # 推論実行
                detections = self.detector.predict(frame)

                # 結果を保存
                result = {
                    "frame_index": frame_count,
                    "detections": detections,
                    "num_detections": len(detections),
                }
                results.append(result)

                detection_count += len(detections)

                # 可視化画像を保存（オプション）
                if len(detections) > 0:
                    vis_image = self.detector.visualize_predictions(frame, detections, show=False)
                    cv2.imwrite(str(output_path / f"frame_{frame_count:06d}.jpg"), vis_image)

            frame_count += 1

        cap.release()

        # 結果を保存
        with open(output_path / "inference_results.json", "w") as f:
            json.dump(results, f, indent=2)

        return {
            "total_frames": frame_count,
            "processed_frames": len(results),
            "total_detections": detection_count,
            "average_detections_per_frame": detection_count / len(results) if results else 0,
        }


def main():
    """メイン関数（テスト用）"""
    import argparse

    parser = argparse.ArgumentParser(description="統合モデル訓練システム")
    parser.add_argument(
        "command", choices=["train", "fine-tune", "inference"], help="実行するコマンド"
    )
    parser.add_argument("--data", required=True, help="データパス")
    parser.add_argument("--model", help="モデルパス")
    parser.add_argument("--output", required=True, help="出力パス")
    parser.add_argument("--config", help="設定ファイル")
    parser.add_argument("--epochs", type=int, default=100, help="エポック数")

    args = parser.parse_args()

    # トレーナーの初期化
    trainer = UnifiedModelTrainer(args.config)

    if args.command == "train":
        # 初期モデルの訓練
        trainer.create_initial_model(args.data, args.output)

    elif args.command == "fine-tune":
        # ファインチューニング
        if not args.model:
            print("エラー: --model オプションが必要です")
            return
        trainer.fine_tune_model(args.model, args.data, args.output, args.epochs)

    elif args.command == "inference":
        # バッチ推論
        if not args.model:
            print("エラー: --model オプションが必要です")
            return
        trainer.batch_inference(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
