"""
動画前処理モジュール

動画からフレーム抽出、前処理、シーン検出などを行う
"""

from pathlib import Path

import cv2
import numpy as np

from ..utils.config import ConfigManager
from ..utils.logger import LoggerMixin
from ..utils.video_codec_validator import VideoCodecValidator


class VideoProcessor(LoggerMixin):
    """動画処理を行うクラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
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
        self.output_format = self.video_config.get("frame_extraction", {}).get(
            "output_format", "jpg"
        )
        self.quality = self.video_config.get("frame_extraction", {}).get("quality", 95)

        self.target_width = self.video_config.get("preprocessing", {}).get("target_width", 1920)
        self.target_height = self.video_config.get("preprocessing", {}).get("target_height", 1080)
        self.normalize = self.video_config.get("preprocessing", {}).get("normalize", True)
        self.denoise = self.video_config.get("preprocessing", {}).get("denoise", True)

        # フレーム類似度チェック設定
        self.skip_similar_frames = self.video_config.get("frame_extraction", {}).get(
            "skip_similar_frames", True
        )
        self.similarity_threshold = self.video_config.get("frame_extraction", {}).get(
            "similarity_threshold", 0.99
        )
        self.prev_frame_hash = None  # 前フレームのハッシュ値

        # コーデックバリデーター
        self.codec_validator = VideoCodecValidator()

        self.logger.info("VideoProcessorが初期化されました")

    def calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        2つのフレームの類似度を計算

        Args:
            frame1: フレーム1
            frame2: フレーム2

        Returns:
            類似度（0.0～1.0）
        """
        # グレースケールに変換
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

        # サイズを統一
        size = (640, 480)  # 計算効率のために縮小
        gray1 = cv2.resize(gray1, size)
        gray2 = cv2.resize(gray2, size)

        # 構造類似度指標（SSIM）の簡易版
        # 平均二乗誤差から類似度を計算
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        max_pixel_value = 255.0

        if mse == 0:
            return 1.0

        # PSNRを類似度に変換
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        # PSNRを 0-1 の範囲に正規化（40dB以上でほぼ同一）
        similarity = min(1.0, psnr / 40.0)

        return similarity

    def calculate_frame_hash(self, frame: np.ndarray) -> str:
        """
        フレームのハッシュ値を計算（高速な重複チェック用）

        Args:
            frame: フレーム

        Returns:
            ハッシュ値
        """
        # グレースケールに変換して縮小
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        small = cv2.resize(gray, (16, 16))

        # 平均値を計算
        avg = small.mean()

        # バイナリハッシュを作成
        diff = small > avg
        return diff.tobytes().hex()

    def extract_frames(
        self, video_path: str, output_dir: str | None = None, frame_skip_manager=None
    ) -> list[str]:
        """
        動画からフレームを抽出

        Args:
            video_path: 動画ファイルのパス
            output_dir: 出力ディレクトリ（指定しない場合はtempディレクトリ）
            frame_skip_manager: フレームスキップマネージャー（対局画面のみ抽出）

        Returns:
            抽出されたフレームファイルのパスリスト
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        # 動画ファイルのコーデックを検証
        self.logger.info("動画ファイルのコーデックを検証中...")
        validation_result = self.codec_validator.validate_video_file(str(video_path))

        if not validation_result["valid"]:
            self.logger.error(f"動画ファイルの検証に失敗: {validation_result}")
            self.logger.error(f"推奨事項: {validation_result['recommendation']}")

            # 自動変換を試みる（オプション）
            if validation_result["codec_info"] and not validation_result["opencv_test"]["can_open"]:
                raise ValueError(
                    f"動画ファイルを開けません。{validation_result['recommendation']}\n"
                    f"検出されたコーデック: {validation_result.get('detected_codec', '不明')}"
                )
        else:
            self.logger.info(
                f"動画ファイル検証成功 - コーデック: {validation_result.get('detected_codec', '不明')}, "
                f"解像度: {validation_result['opencv_test'].get('width')}x{validation_result['opencv_test'].get('height')}"
            )

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

            self.logger.info(
                f"動画情報 - FPS: {video_fps:.2f}, 総フレーム数: {total_frames}, "
                f"長さ: {duration:.2f}秒"
            )

            # フレーム間隔を計算
            frame_interval = int(video_fps / self.fps) if video_fps > 0 else 1

            extracted_files = []
            frame_count = 0
            extracted_count = 0
            prev_frame = None  # 前フレームを保持
            similar_frame_count = 0  # 類似フレーム数

            # 動画IDを取得（フレームスキップマネージャー用）
            video_id = Path(video_path).stem

            while True:
                ret, frame = cap.read()
                if not ret:
                    if frame_count == 0:
                        self.logger.error(
                            "最初のフレームも読み込めません。動画ファイルが破損している可能性があります。"
                        )
                    break

                # フレームの妥当性チェック
                if frame is None or frame.size == 0:
                    self.logger.warning(f"フレーム {frame_count} が無効です。スキップします。")
                    frame_count += 1
                    continue

                # フレームスキップチェック
                if frame_skip_manager and frame_skip_manager.should_skip_frame(
                    video_id, frame_count
                ):
                    frame_count += 1
                    continue

                # 指定間隔でフレームを抽出
                if frame_count % frame_interval == 0:
                    # 類似フレームチェック
                    should_skip = False
                    if self.skip_similar_frames and prev_frame is not None:
                        # ハッシュ値で高速チェック
                        current_hash = self.calculate_frame_hash(frame)
                        if current_hash == self.prev_frame_hash:
                            should_skip = True
                            similar_frame_count += 1
                        else:
                            # ハッシュが異なる場合は詳細な類似度チェック
                            similarity = self.calculate_frame_similarity(frame, prev_frame)
                            if similarity >= self.similarity_threshold:
                                should_skip = True
                                similar_frame_count += 1
                            else:
                                self.prev_frame_hash = current_hash
                    else:
                        self.prev_frame_hash = self.calculate_frame_hash(frame)

                    if should_skip:
                        frame_count += 1
                        continue

                    # フレームを前処理
                    processed_frame = self.preprocess_frame(frame)
                    prev_frame = frame.copy()  # 次回比較用に保存

                    # ファイル名を生成
                    timestamp = frame_count / video_fps if video_fps > 0 else frame_count
                    filename = f"frame_{extracted_count:06d}_{timestamp:.2f}s.{self.output_format}"
                    output_path = output_dir / filename

                    # フレームを保存
                    if self.output_format.lower() in ["jpg", "jpeg"]:
                        cv2.imwrite(
                            str(output_path),
                            processed_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.quality],
                        )
                    else:
                        cv2.imwrite(str(output_path), processed_frame)

                    extracted_files.append(str(output_path))
                    extracted_count += 1

                frame_count += 1

                # 進捗表示
                if frame_count % 1000 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self.logger.info(
                        f"フレーム抽出進捗: {progress:.1f}% ({extracted_count}フレーム抽出)"
                    )

            self.logger.info(
                f"フレーム抽出完了: {extracted_count}フレームを抽出 "
                f"(類似フレーム{similar_frame_count}フレームをスキップ)"
            )
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

            padded[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
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

    def detect_scene_changes(self, video_path: str, threshold: float = 0.3) -> list[float]:
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
                        self.logger.debug(
                            f"シーン変更検出: {timestamp:.2f}秒 (差分率: {diff_ratio:.3f})"
                        )

                prev_frame = gray
                frame_count += 1

            self.logger.info(f"シーン変更検出完了: {len(scene_changes)}箇所")
            return scene_changes

        finally:
            cap.release()

    def filter_relevant_frames(self, frame_paths: list[str]) -> list[str]:
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
        return laplacian_var >= 100  # 閾値は調整が必要

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
                "file_size": video_path.stat().st_size,
            }

            if info["fps"] > 0:
                info["duration"] = info["frame_count"] / info["fps"]

            return info

        finally:
            cap.release()
