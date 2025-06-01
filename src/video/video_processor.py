"""
動画前処理モジュール

動画からフレーム抽出、前処理、シーン検出などを行う
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import ffmpeg

from ..utils.config import ConfigManager
from ..utils.logger import LoggerMixin


class VideoProcessor(LoggerMixin):
    """動画処理を行うクラス"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        動画処理クラスの初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.video_config = self.config_manager.get_video_config()
        self.directories = self.config_manager.get_directories()
        
        # 設定値の取得
        self.fps = self.video_config.get("frame_extraction", {}).get("fps", 1)
        self.output_format = self.video_config.get("frame_extraction", {}).get("output_format", "jpg")
        self.quality = self.video_config.get("frame_extraction", {}).get("quality", 95)
        
        self.target_width = self.video_config.get("preprocessing", {}).get("target_width", 1920)
        self.target_height = self.video_config.get("preprocessing", {}).get("target_height", 1080)
        self.normalize = self.video_config.get("preprocessing", {}).get("normalize", True)
        self.denoise = self.video_config.get("preprocessing", {}).get("denoise", True)
        
        self.logger.info("VideoProcessorが初期化されました")
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        動画からフレームを抽出
        
        Args:
            video_path: 動画ファイルのパス
            output_dir: 出力ディレクトリ（指定しない場合はtempディレクトリ）
            
        Returns:
            抽出されたフレームファイルのパスリスト
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
        if output_dir is None:
            output_dir = Path(self.directories.get("temp", "data/temp")) / "frames"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"フレーム抽出開始: {video_path} -> {output_dir}")
        
        # OpenCVで動画を開く
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")
        
        try:
            # 動画情報を取得
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            self.logger.info(f"動画情報 - FPS: {video_fps:.2f}, 総フレーム数: {total_frames}, 長さ: {duration:.2f}秒")
            
            # フレーム間隔を計算
            frame_interval = int(video_fps / self.fps) if video_fps > 0 else 1
            
            extracted_files = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 指定間隔でフレームを抽出
                if frame_count % frame_interval == 0:
                    # フレームを前処理
                    processed_frame = self.preprocess_frame(frame)
                    
                    # ファイル名を生成
                    timestamp = frame_count / video_fps if video_fps > 0 else frame_count
                    filename = f"frame_{extracted_count:06d}_{timestamp:.2f}s.{self.output_format}"
                    output_path = output_dir / filename
                    
                    # フレームを保存
                    if self.output_format.lower() in ['jpg', 'jpeg']:
                        cv2.imwrite(str(output_path), processed_frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    else:
                        cv2.imwrite(str(output_path), processed_frame)
                    
                    extracted_files.append(str(output_path))
                    extracted_count += 1
                
                frame_count += 1
                
                # 進捗表示
                if frame_count % 1000 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self.logger.info(f"フレーム抽出進捗: {progress:.1f}% ({extracted_count}フレーム抽出)")
            
            self.logger.info(f"フレーム抽出完了: {extracted_count}フレームを抽出")
            return extracted_files
            
        finally:
            cap.release()
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームの前処理
        
        Args:
            frame: 入力フレーム
            
        Returns:
            前処理済みフレーム
        """
        processed = frame.copy()
        
        # リサイズ
        processed = self.resize_frame(processed, self.target_width, self.target_height)
        
        # ノイズ除去
        if self.denoise:
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
        
        # 正規化（必要に応じて）
        if self.normalize:
            processed = self.normalize_frame(processed)
        
        return processed
    
    def resize_frame(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        フレームをリサイズ
        
        Args:
            frame: 入力フレーム
            target_width: 目標幅
            target_height: 目標高さ
            
        Returns:
            リサイズ済みフレーム
        """
        height, width = frame.shape[:2]
        
        # アスペクト比を保持してリサイズ
        aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height
        
        if aspect_ratio > target_aspect_ratio:
            # 幅を基準にリサイズ
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # 高さを基準にリサイズ
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 目標サイズに合わせてパディング
        if new_width != target_width or new_height != target_height:
            # 黒でパディング
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # 中央に配置
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            return padded
        
        return resized
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームの正規化
        
        Args:
            frame: 入力フレーム
            
        Returns:
            正規化済みフレーム
        """
        # ヒストグラム均等化
        if len(frame.shape) == 3:
            # カラー画像の場合、LAB色空間でL成分のみ均等化
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # グレースケール画像の場合
            return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(frame)
    
    def detect_scene_changes(self, video_path: str, threshold: float = 0.3) -> List[float]:
        """
        シーン変更を検出
        
        Args:
            video_path: 動画ファイルのパス
            threshold: シーン変更検出の閾値
            
        Returns:
            シーン変更が発生した時刻のリスト（秒）
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
        self.logger.info(f"シーン変更検出開始: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            scene_changes = []
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # グレースケールに変換
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # フレーム間の差分を計算
                    diff = cv2.absdiff(prev_frame, gray)
                    diff_ratio = np.mean(diff) / 255.0
                    
                    # 閾値を超えた場合はシーン変更とみなす
                    if diff_ratio > threshold:
                        timestamp = frame_count / fps if fps > 0 else frame_count
                        scene_changes.append(timestamp)
                        self.logger.debug(f"シーン変更検出: {timestamp:.2f}秒 (差分率: {diff_ratio:.3f})")
                
                prev_frame = gray
                frame_count += 1
            
            self.logger.info(f"シーン変更検出完了: {len(scene_changes)}箇所")
            return scene_changes
            
        finally:
            cap.release()
    
    def filter_relevant_frames(self, frame_paths: List[str]) -> List[str]:
        """
        麻雀関連フレームを抽出（フェーズ1では基本的なフィルタリングのみ）
        
        Args:
            frame_paths: フレームファイルのパスリスト
            
        Returns:
            フィルタリング済みフレームパスリスト
        """
        self.logger.info(f"関連フレームフィルタリング開始: {len(frame_paths)}フレーム")
        
        # フェーズ1では簡単なフィルタリングのみ実装
        # 将来的にはAIモデルを使用して麻雀関連フレームを判定
        
        filtered_frames = []
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # 基本的な品質チェック
            if self._is_valid_frame(frame):
                filtered_frames.append(frame_path)
        
        self.logger.info(f"関連フレームフィルタリング完了: {len(filtered_frames)}フレーム")
        return filtered_frames
    
    def _is_valid_frame(self, frame: np.ndarray) -> bool:
        """
        フレームの有効性をチェック
        
        Args:
            frame: チェック対象フレーム
            
        Returns:
            有効かどうか
        """
        # 基本的な品質チェック
        if frame is None or frame.size == 0:
            return False
        
        # 極端に暗い/明るいフレームを除外
        mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if mean_brightness < 20 or mean_brightness > 235:
            return False
        
        # ブラー検出（簡易版）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # 閾値は調整が必要
            return False
        
        return True
    
    def get_video_info(self, video_path: str) -> dict:
        """
        動画の基本情報を取得
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            動画情報の辞書
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")
        
        try:
            info = {
                "path": str(video_path),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": 0,
                "file_size": video_path.stat().st_size
            }
            
            if info["fps"] > 0:
                info["duration"] = info["frame_count"] / info["fps"]
            
            return info
            
        finally:
            cap.release()