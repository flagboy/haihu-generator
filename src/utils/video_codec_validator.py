"""
動画コーデック検証モジュール

動画ファイルのコーデックをチェックし、サポート状況を確認する
"""

import subprocess
from pathlib import Path
from typing import Any

import cv2

from .logger import LoggerMixin


class VideoCodecValidator(LoggerMixin):
    """動画コーデックの検証を行うクラス"""

    # サポートされているコーデック（OpenCVで一般的に利用可能）
    SUPPORTED_CODECS = {
        "h264": ["h264", "avc1", "avc", "x264"],
        "h265": ["h265", "hevc", "hvc1", "x265"],
        "mpeg4": ["mp4v", "mpeg4", "xvid", "divx"],
        "mjpeg": ["mjpeg", "mjpg"],
        "vp8": ["vp8", "vp80"],
        "vp9": ["vp9", "vp90"],
        "theora": ["theora", "theo"],
    }

    def __init__(self):
        """初期化"""
        self.logger.info("VideoCodecValidatorが初期化されました")

    def validate_video_file(self, video_path: str) -> dict[str, Any]:
        """
        動画ファイルのコーデックを検証

        Args:
            video_path: 動画ファイルのパス

        Returns:
            検証結果の辞書
        """
        video_path = Path(video_path)

        if not video_path.exists():
            return {
                "valid": False,
                "error": "ファイルが存在しません",
                "codec_info": None,
                "supported": False,
            }

        # OpenCVでの読み込みテスト
        opencv_result = self._test_opencv_compatibility(str(video_path))

        # ffmpegでコーデック情報を取得
        codec_info = self._get_codec_info_ffmpeg(str(video_path))

        # コーデックのサポート状況を確認
        is_supported = False
        detected_codec = None

        if codec_info:
            detected_codec = codec_info.get("codec_name", "").lower()
            is_supported = self._is_codec_supported(detected_codec)

        return {
            "valid": opencv_result["can_open"],
            "opencv_test": opencv_result,
            "codec_info": codec_info,
            "detected_codec": detected_codec,
            "supported": is_supported,
            "recommendation": self._get_recommendation(opencv_result, codec_info, is_supported),
        }

    def _test_opencv_compatibility(self, video_path: str) -> dict[str, Any]:
        """OpenCVでの読み込みテスト"""
        result = {
            "can_open": False,
            "can_read_frames": False,
            "frame_count": 0,
            "error": None,
        }

        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            result["can_open"] = cap.isOpened()

            if result["can_open"]:
                # フレーム読み込みテスト
                ret, frame = cap.read()
                result["can_read_frames"] = ret and frame is not None

                if result["can_read_frames"]:
                    # 総フレーム数を取得
                    result["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    result["fps"] = cap.get(cv2.CAP_PROP_FPS)
                    result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"OpenCV読み込みテストエラー: {e}")
        finally:
            if cap is not None:
                cap.release()

        return result

    def _get_codec_info_ffmpeg(self, video_path: str) -> dict[str, Any] | None:
        """ffmpegを使用してコーデック情報を取得"""
        try:
            # ffprobeを使用して詳細情報を取得
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,codec_long_name,width,height,r_frame_rate,bit_rate",
                "-of",
                "json",
                video_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                import json

                data = json.loads(result.stdout)
                if data.get("streams"):
                    stream = data["streams"][0]
                    return {
                        "codec_name": stream.get("codec_name"),
                        "codec_long_name": stream.get("codec_long_name"),
                        "width": stream.get("width"),
                        "height": stream.get("height"),
                        "frame_rate": stream.get("r_frame_rate"),
                        "bit_rate": stream.get("bit_rate"),
                    }
            else:
                # ffprobeが失敗した場合、ffmpegで簡易的に情報取得
                cmd = ["ffmpeg", "-i", video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                # エラー出力から情報を抽出
                for line in result.stderr.split("\n"):
                    if "Video:" in line:
                        parts = line.split("Video:")[1].split(",")
                        if parts:
                            codec_info = parts[0].strip()
                            return {"codec_name": codec_info, "raw_info": line}

        except subprocess.TimeoutExpired:
            self.logger.warning("ffmpeg/ffprobeタイムアウト")
        except FileNotFoundError:
            self.logger.warning("ffmpeg/ffprobeが見つかりません")
        except Exception as e:
            self.logger.error(f"コーデック情報取得エラー: {e}")

        return None

    def _is_codec_supported(self, codec_name: str) -> bool:
        """コーデックがサポートされているか確認"""
        if not codec_name:
            return False

        codec_lower = codec_name.lower()

        for _codec_family, variants in self.SUPPORTED_CODECS.items():
            if codec_lower in variants:
                return True

        return False

    def _get_recommendation(self, opencv_result: dict, codec_info: dict, is_supported: bool) -> str:
        """推奨事項を生成"""
        if opencv_result["can_read_frames"]:
            return "この動画ファイルは正常に処理できます。"

        if not opencv_result["can_open"]:
            if codec_info:
                codec_name = codec_info.get("codec_name", "不明")
                if not is_supported:
                    return (
                        f"コーデック '{codec_name}' はサポートされていません。"
                        f"H.264, H.265, MPEG4などの一般的なコーデックに変換することを推奨します。"
                    )
                else:
                    return (
                        f"コーデック '{codec_name}' はサポート対象ですが、"
                        f"ファイルが破損している可能性があります。"
                    )
            else:
                return (
                    "動画ファイルの形式が認識できません。ファイルが破損している可能性があります。"
                )

        return "動画ファイルは開けますが、フレームの読み込みに問題があります。"

    def convert_video_safe(
        self, input_path: str, output_path: str, target_codec: str = "h264"
    ) -> tuple[bool, str | None]:
        """
        動画を安全なコーデックに変換

        Args:
            input_path: 入力動画パス
            output_path: 出力動画パス
            target_codec: 変換先コーデック

        Returns:
            (成功フラグ, エラーメッセージ)
        """
        try:
            # コーデック設定
            codec_settings = {
                "h264": ["-c:v", "libx264", "-preset", "fast", "-crf", "23"],
                "h265": ["-c:v", "libx265", "-preset", "fast", "-crf", "28"],
                "mpeg4": ["-c:v", "mpeg4", "-q:v", "3"],
            }

            if target_codec not in codec_settings:
                return False, f"未対応のターゲットコーデック: {target_codec}"

            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                *codec_settings[target_codec],
                "-y",  # 上書き確認なし
                output_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # 変換後のファイルを検証
                validation = self.validate_video_file(output_path)
                if validation["valid"]:
                    return True, None
                else:
                    return False, "変換は完了しましたが、出力ファイルの検証に失敗しました"
            else:
                return False, f"変換エラー: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "変換タイムアウト（5分以上）"
        except Exception as e:
            return False, f"変換中にエラーが発生: {str(e)}"
