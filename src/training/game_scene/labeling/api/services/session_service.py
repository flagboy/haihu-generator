"""
セッション管理サービス

セッションの作成、取得、削除などのビジネスロジックを提供
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ......utils.logger import LoggerMixin
from ...scene_labeling_session import SceneLabelingSession
from ..middleware.error_handler import NotFoundError, ValidationError


class SessionService(LoggerMixin):
    """セッション管理サービス"""

    def __init__(self):
        """初期化"""
        self._sessions: dict[str, SceneLabelingSession] = {}

    def create_session(
        self, video_path: str, session_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        セッションを作成または再開

        Args:
            video_path: 動画ファイルパス
            session_id: 既存セッションID（オプション）
            metadata: メタデータ

        Returns:
            セッション情報

        Raises:
            ValidationError: 動画ファイルが見つからない
            ConflictError: セッションIDが既に使用中
        """
        # 動画ファイルの存在確認
        video_path_obj = self._resolve_video_path(video_path)
        if not video_path_obj.exists():
            raise ValidationError(f"動画ファイルが見つかりません: {video_path}")

        # セッションIDの処理
        if session_id:
            if session_id in self._sessions:
                # 既存セッションを返す
                self.logger.info(f"既存セッションを再開: {session_id}")
                return self._get_session_info(session_id)
        else:
            session_id = str(uuid.uuid4())

        # 新しいセッションを作成
        try:
            session = SceneLabelingSession(
                video_path=str(video_path_obj), session_id=session_id, db_path=None
            )
            self._sessions[session_id] = session
            self.logger.info(f"新しいセッションを作成: {session_id}")

            # メタデータを保存
            if metadata:
                session.metadata = metadata

            return self._get_session_info(session_id)

        except Exception as e:
            self.logger.error(f"セッション作成エラー: {e}")
            raise ValidationError(f"セッションの作成に失敗しました: {str(e)}") from e

    def get_session(self, session_id: str) -> SceneLabelingSession:
        """
        セッションを取得

        Args:
            session_id: セッションID

        Returns:
            セッションインスタンス

        Raises:
            NotFoundError: セッションが見つからない
        """
        if session_id not in self._sessions:
            raise NotFoundError("セッション", session_id)
        return self._sessions[session_id]

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        """
        セッション情報を取得

        Args:
            session_id: セッションID

        Returns:
            セッション情報

        Raises:
            NotFoundError: セッションが見つからない
        """
        if session_id not in self._sessions:
            raise NotFoundError("セッション", session_id)
        return self._get_session_info(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        全セッション一覧を取得

        Returns:
            セッション情報のリスト
        """
        return [self._get_session_info(sid) for sid in self._sessions]

    def delete_session(self, session_id: str) -> bool:
        """
        セッションを削除

        Args:
            session_id: セッションID

        Returns:
            削除成功かどうか

        Raises:
            NotFoundError: セッションが見つからない
        """
        if session_id not in self._sessions:
            raise NotFoundError("セッション", session_id)

        session = self._sessions[session_id]
        session.close()
        del self._sessions[session_id]
        self.logger.info(f"セッションを削除: {session_id}")
        return True

    def close_session(self, session_id: str) -> bool:
        """
        セッションを終了

        Args:
            session_id: セッションID

        Returns:
            終了成功かどうか

        Raises:
            NotFoundError: セッションが見つからない
        """
        session = self.get_session(session_id)
        session.close()
        self.logger.info(f"セッションを終了: {session_id}")
        return True

    def clear_all_sessions(self) -> int:
        """
        全セッションをクリア

        Returns:
            クリアしたセッション数
        """
        count = len(self._sessions)
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self.logger.info(f"{count}個のセッションをクリア")
        return count

    def session_exists(self, session_id: str) -> bool:
        """
        セッションが存在するか確認

        Args:
            session_id: セッションID

        Returns:
            存在するかどうか
        """
        return session_id in self._sessions

    def _get_session_info(self, session_id: str) -> dict[str, Any]:
        """セッション情報を取得（内部用）"""
        session = self._sessions[session_id]

        # ラベル付きフレーム数を計算
        labeled_count = sum(1 for frame in session.get_all_frames() if frame.get("label"))

        return {
            "session_id": session_id,
            "video_path": session.video_path,
            "total_frames": session.total_frames,
            "labeled_frames": labeled_count,
            "created_at": session.created_at.isoformat()
            if hasattr(session, "created_at")
            else datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active"
            if not hasattr(session, "is_closed") or not session.is_closed()
            else "closed",
            "metadata": getattr(session, "metadata", {}),
        }

    def _resolve_video_path(self, video_path: str) -> Path:
        """動画パスを解決"""
        video_path_obj = Path(video_path)
        if video_path_obj.exists():
            return video_path_obj

        # プロジェクトルートからの相対パスとして試す
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent

        possible_paths = [
            project_root / video_path,
            project_root / "web_interface" / video_path,
            project_root / "web_interface" / "web_interface" / video_path,
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return video_path_obj  # 見つからない場合は元のパスを返す
