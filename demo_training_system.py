"""
教師データ作成システムのデモンストレーション

フェーズ1で実装した基盤システムの動作確認を行う
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.training.annotation_data import AnnotationData, FrameAnnotation, TileAnnotation, BoundingBox
from src.training.dataset_manager import DatasetManager
from src.training.frame_extractor import FrameExtractor, FrameQualityAnalyzer
from src.training.semi_auto_labeler import SemiAutoLabeler
from src.utils.config import ConfigManager
from src.utils.logger import get_logger


def create_sample_video(output_path: str, duration: int = 10, fps: int = 30):
    """
    デモ用のサンプル動画を作成
    
    Args:
        output_path: 出力パス
        duration: 動画の長さ（秒）
        fps: フレームレート
    """
    logger = get_logger(__name__)
    logger.info(f"サンプル動画を作成中: {output_path}")
    
    # 動画ライターを初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # 背景を作成（緑のテーブル風）
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        frame[:, :, 1] = 100  # 緑っぽく
        
        # 時間に応じて変化する要素を追加
        time_factor = frame_num / total_frames
        
        # 手牌エリア（画面下部）
        hand_area = (50, 400, 590, 450)
        cv2.rectangle(frame, (hand_area[0], hand_area[1]), (hand_area[2], hand_area[3]), (80, 120, 80), -1)
        
        # 捨て牌エリア（画面中央）
        discard_area = (200, 200, 440, 350)
        cv2.rectangle(frame, (discard_area[0], discard_area[1]), (discard_area[2], discard_area[3]), (60, 100, 60), -1)
        
        # 模擬的な牌を描画
        for i in range(13):  # 手牌13枚
            x = hand_area[0] + i * 40
            y = hand_area[1] + 5
            # 牌の形状（白い矩形）
            cv2.rectangle(frame, (x, y), (x + 35, y + 40), (240, 240, 240), -1)
            cv2.rectangle(frame, (x, y), (x + 35, y + 40), (0, 0, 0), 2)
            
            # 牌の文字（簡易的）
            tile_char = str((i % 9) + 1)
            cv2.putText(frame, tile_char, (x + 12, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 捨て牌（時間とともに増加）
        discard_count = int(time_factor * 20)
        for i in range(min(discard_count, 20)):
            row = i // 6
            col = i % 6
            x = discard_area[0] + col * 35
            y = discard_area[1] + row * 25
            
            cv2.rectangle(frame, (x, y), (x + 30, y + 20), (220, 220, 220), -1)
            cv2.rectangle(frame, (x, y), (x + 30, y + 20), (0, 0, 0), 1)
        
        # フレーム番号を表示
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ノイズを追加（リアルさのため）
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    logger.info(f"サンプル動画作成完了: {output_path}")


def demo_annotation_data():
    """アノテーションデータのデモ"""
    logger = get_logger(__name__)
    logger.info("=== アノテーションデータのデモ ===")
    
    # アノテーションデータを作成
    annotation_data = AnnotationData()
    
    # 動画アノテーションを作成
    video_info = {
        "duration": 10.0,
        "fps": 30.0,
        "width": 640,
        "height": 480
    }
    video_id = annotation_data.create_video_annotation("demo_video.mp4", video_info)
    logger.info(f"動画アノテーション作成: {video_id}")
    
    # フレームアノテーションを作成
    for i in range(5):
        # サンプル牌アノテーション
        tiles = []
        for j in range(3):
            bbox = BoundingBox(50 + j * 40, 400, 85 + j * 40, 440)
            tile = TileAnnotation(
                tile_id=f"{j+1}m",
                bbox=bbox,
                confidence=0.9 - j * 0.1,
                area_type="hand",
                annotator="demo"
            )
            tiles.append(tile)
        
        frame_annotation = FrameAnnotation(
            frame_id=f"demo_frame_{i:03d}",
            image_path=f"demo_frame_{i:03d}.jpg",
            image_width=640,
            image_height=480,
            timestamp=i * 2.0,
            tiles=tiles,
            quality_score=0.8 + i * 0.05,
            annotator="demo"
        )
        
        annotation_data.add_frame_annotation(video_id, frame_annotation)
    
    # 統計情報を表示
    stats = annotation_data.get_all_statistics()
    logger.info(f"統計情報: {stats}")
    
    # JSON保存
    output_path = "demo_annotations.json"
    if annotation_data.save_to_json(output_path):
        logger.info(f"アノテーションデータを保存: {output_path}")
    
    return annotation_data


def demo_dataset_manager():
    """データセット管理のデモ"""
    logger = get_logger(__name__)
    logger.info("=== データセット管理のデモ ===")
    
    # 設定管理
    config_manager = ConfigManager()
    
    # データセット管理を初期化
    dataset_manager = DatasetManager(config_manager)
    
    # アノテーションデータを作成
    annotation_data = demo_annotation_data()
    
    # データベースに保存
    if dataset_manager.save_annotation_data(annotation_data):
        logger.info("アノテーションデータをデータベースに保存")
    
    # データベースから読み込み
    loaded_data = dataset_manager.load_annotation_data()
    logger.info(f"読み込み完了: {len(loaded_data.video_annotations)}動画")
    
    # データセットバージョンを作成
    version_id = dataset_manager.create_dataset_version(
        annotation_data, 
        "v1.0.0", 
        "初回デモバージョン"
    )
    if version_id:
        logger.info(f"データセットバージョン作成: {version_id}")
    
    # 統計情報を表示
    stats = dataset_manager.get_dataset_statistics()
    logger.info(f"データセット統計: {stats}")
    
    return dataset_manager


def demo_frame_extractor():
    """フレーム抽出のデモ"""
    logger = get_logger(__name__)
    logger.info("=== フレーム抽出のデモ ===")
    
    # サンプル動画を作成
    video_path = "demo_video.mp4"
    if not Path(video_path).exists():
        create_sample_video(video_path)
    
    # 設定管理
    config_manager = ConfigManager()
    
    # フレーム抽出器を初期化
    frame_extractor = FrameExtractor(config_manager)
    
    # 品質分析器のテスト
    quality_analyzer = FrameQualityAnalyzer()
    
    # サンプル画像で品質分析
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    quality_scores = quality_analyzer.analyze_frame_quality(test_image)
    logger.info(f"品質スコア例: {quality_scores}")
    
    # 動画のシーン分析
    try:
        scenes = frame_extractor.analyze_video_scenes(video_path)
        logger.info(f"検出されたシーン数: {len(scenes)}")
        for i, scene in enumerate(scenes[:3]):  # 最初の3シーンを表示
            logger.info(f"シーン {i+1}: {scene}")
    except Exception as e:
        logger.warning(f"シーン分析エラー: {e}")
    
    # フレーム抽出（実際の動画処理は時間がかかるため、統計情報のみ表示）
    stats = frame_extractor.get_extraction_statistics()
    logger.info(f"抽出統計: {stats}")
    
    return frame_extractor


def demo_semi_auto_labeler():
    """半自動ラベリングのデモ"""
    logger = get_logger(__name__)
    logger.info("=== 半自動ラベリングのデモ ===")
    
    # 設定管理
    config_manager = ConfigManager()
    
    try:
        # 半自動ラベラーを初期化（モデルが存在しない場合はエラーになる可能性）
        semi_auto_labeler = SemiAutoLabeler(config_manager)
        
        # テスト用フレームアノテーション
        frame_annotation = FrameAnnotation(
            frame_id="test_frame",
            image_path="test_image.jpg",
            image_width=640,
            image_height=480,
            timestamp=1.0,
            tiles=[]
        )
        
        # テスト画像を作成
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", test_image)
        
        # 遮蔽検出のテスト
        is_occluded, occlusion_ratio = semi_auto_labeler._detect_occlusion(test_image)
        logger.info(f"遮蔽検出結果: 遮蔽={is_occluded}, 遮蔽率={occlusion_ratio:.3f}")
        
        # 修正テンプレートの作成テスト
        from src.training.semi_auto_labeler import PredictionResult
        prediction_result = PredictionResult(frame_annotation, [], [])
        template = semi_auto_labeler.create_correction_template(prediction_result)
        logger.info(f"修正テンプレート作成: {template['frame_id']}")
        
        # 統計情報
        stats = semi_auto_labeler.get_labeling_statistics()
        logger.info(f"ラベリング統計: {stats}")
        
        # クリーンアップ
        Path("test_image.jpg").unlink(missing_ok=True)
        
    except Exception as e:
        logger.warning(f"半自動ラベリングデモでエラー（モデル未学習のため正常）: {e}")
    
    return None


def demo_integration():
    """統合デモ"""
    logger = get_logger(__name__)
    logger.info("=== 統合デモ ===")
    
    try:
        # 1. アノテーションデータの作成
        logger.info("1. アノテーションデータ作成")
        annotation_data = demo_annotation_data()
        
        # 2. データセット管理
        logger.info("2. データセット管理")
        dataset_manager = demo_dataset_manager()
        
        # 3. フレーム抽出
        logger.info("3. フレーム抽出")
        frame_extractor = demo_frame_extractor()
        
        # 4. 半自動ラベリング
        logger.info("4. 半自動ラベリング")
        semi_auto_labeler = demo_semi_auto_labeler()
        
        # 5. 全体統計
        logger.info("5. 全体統計")
        overall_stats = {
            "annotation_data": annotation_data.get_all_statistics(),
            "dataset_manager": dataset_manager.get_dataset_statistics(),
            "frame_extractor": frame_extractor.get_extraction_statistics()
        }
        
        logger.info("=== 統合デモ完了 ===")
        logger.info(f"全体統計: {overall_stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"統合デモでエラー: {e}")
        return False


def main():
    """メイン関数"""
    logger = get_logger(__name__)
    logger.info("教師データ作成システム デモンストレーション開始")
    
    try:
        # 各コンポーネントのデモを実行
        demo_integration()
        
        logger.info("デモンストレーション完了")
        
        # クリーンアップ
        cleanup_files = [
            "demo_video.mp4",
            "demo_annotations.json"
        ]
        
        for file_path in cleanup_files:
            Path(file_path).unlink(missing_ok=True)
        
        logger.info("クリーンアップ完了")
        
    except Exception as e:
        logger.error(f"デモ実行中にエラーが発生: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)