"""
フレームスキップ管理

対局シーン情報を使用して効率的なフレーム処理を実現
"""

import sqlite3
from pathlib import Path

from ....utils.cache_manager import CacheManager
from ....utils.logger import LoggerMixin


class FrameSkipManager(LoggerMixin):
    """フレームスキップ管理クラス"""

    def __init__(
        self, db_path: str = "data/training/game_scene_labels.db", cache_enabled: bool = True
    ):
        """
        初期化

        Args:
            db_path: データベースパス
            cache_enabled: キャッシュを使用するか
        """
        self.db_path = db_path
        self.cache_enabled = cache_enabled

        # キャッシュマネージャー
        if cache_enabled:
            self.cache_manager = CacheManager()

        # メモリキャッシュ
        self._segments_cache: dict[str, list[tuple[int, int]]] = {}
        self._frame_cache: dict[str, dict[int, bool]] = {}

        self.logger.info("FrameSkipManager初期化完了")

    def should_skip_frame(self, video_id: str, frame_number: int) -> bool:
        """
        フレームをスキップすべきか判定

        Args:
            video_id: 動画ID
            frame_number: フレーム番号

        Returns:
            スキップすべきならTrue（非対局シーン）
        """
        # キャッシュから確認
        if video_id in self._frame_cache and frame_number in self._frame_cache[video_id]:
            return not self._frame_cache[video_id][frame_number]

        # セグメント情報から判定
        game_segments = self.get_game_segments(video_id)

        for start, end in game_segments:
            if start <= frame_number <= end:
                # 対局シーンなのでスキップしない
                self._cache_frame_result(video_id, frame_number, True)
                return False

        # 非対局シーンなのでスキップ
        self._cache_frame_result(video_id, frame_number, False)
        return True

    def get_game_segments(self, video_id: str) -> list[tuple[int, int]]:
        """
        対局シーンのセグメントを取得

        Args:
            video_id: 動画ID

        Returns:
            (開始フレーム, 終了フレーム)のリスト
        """
        # メモリキャッシュを確認
        if video_id in self._segments_cache:
            return self._segments_cache[video_id]

        # ファイルキャッシュを確認
        if self.cache_enabled:
            cache_key = f"game_segments_{video_id}"
            cached = self.cache_manager.get(cache_key)
            if cached is not None:
                self._segments_cache[video_id] = cached
                return cached

        # データベースから取得
        segments = self._load_segments_from_db(video_id)

        # キャッシュに保存
        self._segments_cache[video_id] = segments
        if self.cache_enabled:
            self.cache_manager.set(f"game_segments_{video_id}", segments, ttl=3600)

        return segments

    def _load_segments_from_db(self, video_id: str) -> list[tuple[int, int]]:
        """データベースからセグメントを読み込み"""
        if not Path(self.db_path).exists():
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # セグメントテーブルから取得
            cursor.execute(
                """
                SELECT start_frame, end_frame
                FROM game_scene_segments
                WHERE video_id = ? AND scene_type = 'game'
                ORDER BY start_frame
            """,
                (video_id,),
            )

            segments = [(row[0], row[1]) for row in cursor.fetchall()]

            # セグメントがない場合はラベルから生成
            if not segments:
                segments = self._generate_segments_from_labels(cursor, video_id)

            return segments

        finally:
            conn.close()

    def _generate_segments_from_labels(
        self, cursor: sqlite3.Cursor, video_id: str
    ) -> list[tuple[int, int]]:
        """ラベルからセグメントを生成"""
        cursor.execute(
            """
            SELECT frame_number, is_game_scene
            FROM game_scene_labels
            WHERE video_id = ?
            ORDER BY frame_number
        """,
            (video_id,),
        )

        labels = cursor.fetchall()
        if not labels:
            return []

        segments = []
        current_start = None

        for frame_number, is_game_scene in labels:
            if is_game_scene and current_start is None:
                # 対局シーン開始
                current_start = frame_number
            elif not is_game_scene and current_start is not None:
                # 対局シーン終了
                segments.append((current_start, frame_number - 1))
                current_start = None

        # 最後のセグメント
        if current_start is not None:
            segments.append((current_start, labels[-1][0]))

        return segments

    def _cache_frame_result(self, video_id: str, frame_number: int, is_game: bool):
        """フレーム判定結果をキャッシュ"""
        if video_id not in self._frame_cache:
            self._frame_cache[video_id] = {}

        self._frame_cache[video_id][frame_number] = is_game

        # メモリ使用量を制限
        if len(self._frame_cache[video_id]) > 10000:
            # 古いエントリを削除
            sorted_frames = sorted(self._frame_cache[video_id].keys())
            for frame in sorted_frames[:5000]:
                del self._frame_cache[video_id][frame]

    def get_next_game_frame(self, video_id: str, current_frame: int) -> int | None:
        """
        次の対局フレームを取得

        Args:
            video_id: 動画ID
            current_frame: 現在のフレーム番号

        Returns:
            次の対局フレーム番号（ない場合はNone）
        """
        segments = self.get_game_segments(video_id)

        for start, end in segments:
            if current_frame < start:
                return start
            elif start <= current_frame <= end:
                return current_frame + 1 if current_frame < end else None

        return None

    def get_frame_skip_stats(self, video_id: str) -> dict[str, any]:
        """
        フレームスキップ統計を取得

        Args:
            video_id: 動画ID

        Returns:
            統計情報
        """
        segments = self.get_game_segments(video_id)

        if not segments:
            return {"total_segments": 0, "total_game_frames": 0, "skip_ratio": 0.0}

        total_game_frames = sum(end - start + 1 for start, end in segments)

        # 総フレーム数を推定（最後のセグメントの終了フレーム）
        estimated_total = max(end for _, end in segments) if segments else 0

        return {
            "total_segments": len(segments),
            "total_game_frames": total_game_frames,
            "estimated_total_frames": estimated_total,
            "skip_ratio": 1.0 - (total_game_frames / estimated_total if estimated_total > 0 else 0),
            "segments": [(s, e, e - s + 1) for s, e in segments],  # 開始、終了、長さ
        }

    def clear_cache(self, video_id: str | None = None):
        """
        キャッシュをクリア

        Args:
            video_id: 特定の動画のみクリア（Noneで全て）
        """
        if video_id:
            # 特定の動画のキャッシュをクリア
            if video_id in self._segments_cache:
                del self._segments_cache[video_id]
            if video_id in self._frame_cache:
                del self._frame_cache[video_id]

            if self.cache_enabled:
                self.cache_manager.delete(f"game_segments_{video_id}")
        else:
            # 全てクリア
            self._segments_cache.clear()
            self._frame_cache.clear()

            if self.cache_enabled:
                # キャッシュマネージャーの全削除は実装に依存
                pass

    def batch_check_frames(self, video_id: str, frame_numbers: list[int]) -> dict[int, bool]:
        """
        複数フレームを一括チェック

        Args:
            video_id: 動画ID
            frame_numbers: フレーム番号のリスト

        Returns:
            {frame_number: is_game_scene}の辞書
        """
        segments = self.get_game_segments(video_id)
        results = {}

        # セグメントをソート（効率的な検索のため）
        sorted_segments = sorted(segments, key=lambda s: s[0])

        for frame_num in frame_numbers:
            is_game = False

            # 二分探索で該当セグメントを検索
            for start, end in sorted_segments:
                if frame_num < start:
                    break
                if start <= frame_num <= end:
                    is_game = True
                    break

            results[frame_num] = is_game
            self._cache_frame_result(video_id, frame_num, is_game)

        return results
