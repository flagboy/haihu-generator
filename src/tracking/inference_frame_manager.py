"""
推測フレーム管理モジュール

推測されたアクションに関連するフレームを保存し、
後で人間が確認・修正できるようにする。
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.logger import LoggerMixin


@dataclass
class InferenceFrame:
    """推測フレーム情報"""

    frame_id: str  # ユニークID
    frame_number: int
    turn_number: int
    player_index: int
    action_type: str
    inferred_tile: str | None
    confidence: float
    reason: str
    prev_hand: list[str]
    curr_hand: list[str]
    image_path: str
    metadata: dict[str, Any] | None = None
    human_verified: bool = False
    human_correction: dict[str, Any] | None = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class InferenceFrameManager(LoggerMixin):
    """推測フレーム管理クラス"""

    def __init__(self, base_dir: str = "inference_frames"):
        """
        初期化

        Args:
            base_dir: フレーム保存用のベースディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # セッションディレクトリを作成
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)

        # 画像とメタデータ保存用ディレクトリ
        self.images_dir = self.session_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # 推測フレームのインデックス
        self.frames_index: list[InferenceFrame] = []

        self.logger.info(f"InferenceFrameManager初期化: {self.session_dir}")

    def save_inference_frame(
        self,
        frame: np.ndarray | None,
        frame_number: int,
        turn_number: int,
        player_index: int,
        action_type: str,
        inferred_tile: str | None,
        confidence: float,
        reason: str,
        prev_hand: list[str],
        curr_hand: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        推測フレームを保存

        Args:
            frame: フレーム画像（OpenCV形式）
            frame_number: フレーム番号
            turn_number: 巡番号
            player_index: プレイヤー番号
            action_type: アクションタイプ
            inferred_tile: 推測された牌
            confidence: 信頼度
            reason: 推測理由
            prev_hand: 前巡の手牌
            curr_hand: 現在の手牌
            metadata: 追加メタデータ

        Returns:
            フレームID
        """
        # フレームIDを生成（アクションタイプも含めてユニークにする）
        action_suffix = action_type[:3] if action_type else "unk"
        frame_id = (
            f"{self.session_id}_f{frame_number}_t{turn_number}_p{player_index}_{action_suffix}"
        )

        # 画像を保存
        image_filename = f"{frame_id}.jpg"
        image_path = self.images_dir / image_filename

        if frame is not None:
            # 推測情報を画像に描画
            annotated_frame = self._annotate_frame(
                frame.copy(),
                action_type,
                inferred_tile,
                confidence,
                prev_hand,
                curr_hand,
            )
            cv2.imwrite(str(image_path), annotated_frame)
        else:
            # フレームがない場合はプレースホルダー画像を作成
            placeholder = self._create_placeholder_image(
                action_type, inferred_tile, confidence, prev_hand, curr_hand
            )
            cv2.imwrite(str(image_path), placeholder)

        # フレーム情報を作成
        inference_frame = InferenceFrame(
            frame_id=frame_id,
            frame_number=frame_number,
            turn_number=turn_number,
            player_index=player_index,
            action_type=action_type,
            inferred_tile=inferred_tile,
            confidence=confidence,
            reason=reason,
            prev_hand=prev_hand,
            curr_hand=curr_hand,
            image_path=str(image_path),
            metadata=metadata,
        )

        # インデックスに追加
        self.frames_index.append(inference_frame)

        # インデックスを保存
        self._save_index()

        self.logger.info(
            f"推測フレーム保存: {frame_id} - {action_type} "
            f"(牌: {inferred_tile}, 信頼度: {confidence:.2f})"
        )

        return frame_id

    def _annotate_frame(
        self,
        frame: np.ndarray,
        action_type: str,
        inferred_tile: str | None,
        confidence: float,
        prev_hand: list[str],
        curr_hand: list[str],
    ) -> np.ndarray:
        """
        フレームに推測情報を描画

        Args:
            frame: 元のフレーム
            action_type: アクションタイプ
            inferred_tile: 推測された牌
            confidence: 信頼度
            prev_hand: 前巡の手牌
            curr_hand: 現在の手牌

        Returns:
            注釈付きフレーム
        """
        # フォント設定
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # 上部に推測情報を表示
        y_offset = 30
        cv2.putText(
            frame,
            f"INFERRED: {action_type}",
            (10, y_offset),
            font,
            font_scale,
            (0, 0, 255),  # 赤色
            thickness,
        )

        y_offset += 30
        if inferred_tile:
            cv2.putText(
                frame,
                f"Tile: {inferred_tile}",
                (10, y_offset),
                font,
                font_scale,
                (0, 255, 0),  # 緑色
                thickness,
            )
            y_offset += 30

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (10, y_offset),
            font,
            font_scale,
            (255, 255, 0),  # 黄色
            thickness,
        )

        # 下部に手牌の変化を表示
        h, w = frame.shape[:2]
        y_offset = h - 60
        cv2.putText(
            frame,
            f"Prev: {' '.join(prev_hand[:5])}...",
            (10, y_offset),
            font,
            font_scale * 0.8,
            (255, 255, 255),
            thickness - 1,
        )

        y_offset += 25
        cv2.putText(
            frame,
            f"Curr: {' '.join(curr_hand[:5])}...",
            (10, y_offset),
            font,
            font_scale * 0.8,
            (255, 255, 255),
            thickness - 1,
        )

        return frame

    def _create_placeholder_image(
        self,
        action_type: str,
        inferred_tile: str | None,
        confidence: float,
        prev_hand: list[str],
        curr_hand: list[str],
    ) -> np.ndarray:
        """
        プレースホルダー画像を作成

        Args:
            action_type: アクションタイプ
            inferred_tile: 推測された牌
            confidence: 信頼度
            prev_hand: 前巡の手牌
            curr_hand: 現在の手牌

        Returns:
            プレースホルダー画像
        """
        # 黒い背景画像を作成
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # テキストを描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)

        y_offset = 100
        cv2.putText(img, "Frame Not Available", (150, y_offset), font, font_scale, color, thickness)

        y_offset += 50
        cv2.putText(
            img, f"Action: {action_type}", (50, y_offset), font, font_scale, color, thickness
        )

        if inferred_tile:
            y_offset += 40
            cv2.putText(
                img, f"Tile: {inferred_tile}", (50, y_offset), font, font_scale, color, thickness
            )

        y_offset += 40
        cv2.putText(
            img, f"Confidence: {confidence:.2f}", (50, y_offset), font, font_scale, color, thickness
        )

        y_offset += 60
        cv2.putText(img, "Previous hand:", (50, y_offset), font, font_scale * 0.7, color, thickness)
        y_offset += 30
        cv2.putText(
            img, " ".join(prev_hand), (50, y_offset), font, font_scale * 0.6, color, thickness - 1
        )

        y_offset += 40
        cv2.putText(img, "Current hand:", (50, y_offset), font, font_scale * 0.7, color, thickness)
        y_offset += 30
        cv2.putText(
            img, " ".join(curr_hand), (50, y_offset), font, font_scale * 0.6, color, thickness - 1
        )

        return img

    def _save_index(self):
        """インデックスをJSONファイルに保存"""
        index_path = self.session_dir / "index.json"
        index_data = [asdict(frame) for frame in self.frames_index]

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str):
        """
        既存のセッションを読み込み

        Args:
            session_id: セッションID
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"セッションが見つかりません: {session_id}")

        self.session_id = session_id
        self.session_dir = session_dir
        self.images_dir = session_dir / "images"

        # インデックスを読み込み
        index_path = session_dir / "index.json"
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                index_data = json.load(f)
                self.frames_index = [InferenceFrame(**frame) for frame in index_data]

        self.logger.info(f"セッション読み込み完了: {session_id} ({len(self.frames_index)}フレーム)")

    def get_unverified_frames(self) -> list[InferenceFrame]:
        """未検証のフレームを取得"""
        return [frame for frame in self.frames_index if not frame.human_verified]

    def update_frame_correction(
        self, frame_id: str, correction: dict[str, Any], verified: bool = True
    ):
        """
        フレームの修正情報を更新

        Args:
            frame_id: フレームID
            correction: 修正情報
            verified: 検証済みフラグ
        """
        for frame in self.frames_index:
            if frame.frame_id == frame_id:
                frame.human_correction = correction
                frame.human_verified = verified
                self._save_index()
                self.logger.info(f"フレーム修正更新: {frame_id}")
                return

        raise ValueError(f"フレームが見つかりません: {frame_id}")

    def export_corrections(self) -> dict[str, Any]:
        """
        修正情報をエクスポート

        Returns:
            修正情報の辞書
        """
        corrections = {}

        for frame in self.frames_index:
            if frame.human_correction:
                corrections[frame.frame_id] = {
                    "original": {
                        "action_type": frame.action_type,
                        "tile": frame.inferred_tile,
                        "confidence": frame.confidence,
                    },
                    "correction": frame.human_correction,
                    "frame_info": {
                        "frame_number": frame.frame_number,
                        "turn_number": frame.turn_number,
                        "player_index": frame.player_index,
                    },
                }

        return corrections

    def generate_review_html(self) -> str:
        """
        レビュー用のHTMLを生成

        Returns:
            HTMLファイルのパス
        """
        html_path = self.session_dir / "review.html"

        # HTMLの各部分を個別に作成
        html_parts = []

        # ヘッダー部分
        html_parts.append(f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>推測フレームレビュー</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .frame-container {{ border: 1px solid #ccc; margin: 20px 0; padding: 15px; }}
        .frame-image {{ max-width: 640px; margin: 10px 0; }}
        .frame-info {{ margin: 10px 0; }}
        .correction-form {{ margin: 15px 0; padding: 10px; background: #f0f0f0; }}
        .verified {{ background-color: #d4edda; }}
        .unverified {{ background-color: #f8d7da; }}
        button {{ margin: 5px; padding: 5px 15px; }}
    </style>
</head>
<body>
    <h1>推測フレームレビュー - セッション: {self.session_id}</h1>
    <p>合計フレーム数: {len(self.frames_index)} (未検証: {len(self.get_unverified_frames())})</p>
    <hr>
""")

        # 各フレームの情報
        for frame in self.frames_index:
            status_class = "verified" if frame.human_verified else "unverified"
            html_parts.append(f"""
    <div class="frame-container {status_class}">
        <h3>フレーム: {frame.frame_id}</h3>
        <img src="images/{os.path.basename(frame.image_path)}" class="frame-image">
        <div class="frame-info">
            <p><strong>フレーム番号:</strong> {frame.frame_number}</p>
            <p><strong>巡番号:</strong> {frame.turn_number}</p>
            <p><strong>プレイヤー:</strong> {frame.player_index}</p>
            <p><strong>推測アクション:</strong> {frame.action_type}</p>
            <p><strong>推測牌:</strong> {frame.inferred_tile or "なし"}</p>
            <p><strong>信頼度:</strong> {frame.confidence:.2f}</p>
            <p><strong>理由:</strong> {frame.reason}</p>
            <p><strong>前巡の手牌:</strong> {" ".join(frame.prev_hand)}</p>
            <p><strong>現在の手牌:</strong> {" ".join(frame.curr_hand)}</p>
        </div>
        <div class="correction-form">
            <h4>修正フォーム</h4>
            <form id="form_{frame.frame_id}">
                <label>アクションタイプ:
                    <select name="action_type">
                        <option value="">変更なし</option>
                        <option value="draw">ツモ</option>
                        <option value="discard">捨て牌</option>
                        <option value="pon">ポン</option>
                        <option value="chi">チー</option>
                        <option value="kan">カン</option>
                        <option value="reach">リーチ</option>
                        <option value="none">アクションなし</option>
                    </select>
                </label><br>
                <label>牌:
                    <input type="text" name="tile" placeholder="例: 1m, 5p, 7s">
                </label><br>
                <label>コメント:
                    <textarea name="comment" rows="3" cols="50"></textarea>
                </label><br>
                <button type="button" onclick="saveCorrection('{frame.frame_id}')">保存</button>
                <button type="button" onclick="markVerified('{frame.frame_id}')">検証済みにする</button>
            </form>
        </div>
    </div>
""")

        # フッター部分
        html_parts.append("""
    <script>
        function saveCorrection(frameId) {
            // 実際の実装では、サーバーAPIを呼び出して保存
            alert('修正を保存しました: ' + frameId);
        }

        function markVerified(frameId) {
            // 実際の実装では、サーバーAPIを呼び出して検証済みにする
            alert('検証済みにしました: ' + frameId);
        }
    </script>
</body>
</html>
""")

        # HTMLを結合して保存
        html_content = "".join(html_parts)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"レビューHTML生成完了: {html_path}")
        return str(html_path)

    def cleanup_old_sessions(self, days: int = 7):
        """
        古いセッションを削除

        Args:
            days: 保持する日数
        """
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)

        for session_dir in self.base_dir.iterdir():
            if (
                session_dir.is_dir()
                and session_dir != self.session_dir
                and session_dir.stat().st_mtime < cutoff_date
            ):
                shutil.rmtree(session_dir)
                self.logger.info(f"古いセッション削除: {session_dir.name}")

    def get_statistics(self) -> dict[str, Any]:
        """統計情報を取得"""
        total_frames = len(self.frames_index)
        verified_frames = len([f for f in self.frames_index if f.human_verified])
        corrected_frames = len([f for f in self.frames_index if f.human_correction])

        action_types = {}
        for frame in self.frames_index:
            action_types[frame.action_type] = action_types.get(frame.action_type, 0) + 1

        return {
            "session_id": self.session_id,
            "total_frames": total_frames,
            "verified_frames": verified_frames,
            "corrected_frames": corrected_frames,
            "verification_rate": verified_frames / max(1, total_frames),
            "correction_rate": corrected_frames / max(1, total_frames),
            "action_type_distribution": action_types,
        }
