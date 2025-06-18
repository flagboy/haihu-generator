"""
対局画面ラベリングAPIルート
"""

import base64
import sqlite3
import uuid
from pathlib import Path

import cv2
from flask import Blueprint, jsonify, request

from .....utils.logger import LoggerMixin, get_logger
from ...core.game_scene_classifier import GameSceneClassifier
from ..scene_labeling_session import SceneLabelingSession

# ブループリント定義
scene_labeling_bp = Blueprint("scene_labeling", __name__, url_prefix="/api/scene_labeling")

# グローバル変数でセッションを管理
_sessions: dict[str, SceneLabelingSession] = {}
_classifier = None

# モジュールレベルのロガー
_logger = get_logger(__name__)


class SceneLabelingAPI(LoggerMixin):
    """対局画面ラベリングAPI"""

    @classmethod
    def get_classifier(cls) -> GameSceneClassifier:
        """分類器を取得（シングルトン）"""
        global _classifier
        if _classifier is None:
            # TODO: 学習済みモデルパスを設定から取得
            model_path = "models/game_scene_classifier.pth"
            if Path(model_path).exists():
                _classifier = GameSceneClassifier(model_path=model_path)
            else:
                _classifier = GameSceneClassifier()
        return _classifier


@scene_labeling_bp.route("/sessions/clear", methods=["POST"])
def clear_sessions():
    """全セッションをクリア"""
    global _sessions
    count = len(_sessions)
    _sessions.clear()
    _logger.info(f"{count}個のセッションをクリアしました")
    return jsonify({"success": True, "cleared_count": count})


@scene_labeling_bp.route("/sessions", methods=["POST"])
def create_session():
    """ラベリングセッションを作成または再開"""
    logger = get_logger(__name__)

    try:
        logger.info("セッション作成リクエストを受信")
        data = request.get_json()
        logger.info(f"リクエストデータ: {data}")

        video_path = data.get("video_path")
        session_id = data.get("session_id")  # 既存セッションIDが指定された場合

        # 動画パスの処理
        if not video_path:
            return jsonify({"error": "動画パスが指定されていません"}), 400

        # 相対パスと絶対パスの両方を試す
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            # プロジェクトルートからの相対パスとして試す
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent

            # いくつかのパターンを試す
            possible_paths = [
                project_root / video_path,  # プロジェクトルートからの相対パス
                project_root
                / "web_interface"
                / video_path,  # web_interface/web_interface/uploads/...の場合
                Path(
                    video_path.replace(
                        "web_interface/uploads/", "web_interface/web_interface/uploads/"
                    )
                ),  # パスの修正
            ]

            found = False
            for possible_path in possible_paths:
                if possible_path.exists():
                    video_path = str(possible_path)
                    video_path_obj = possible_path
                    found = True
                    break

            if not found:
                return jsonify({"error": f"動画ファイルが見つかりません: {video_path}"}), 400

        # ビデオIDを取得（ファイル名から拡張子を除いたもの）
        video_id = Path(video_path).stem
        logger.info(f"ビデオID: {video_id}")

        # 同じビデオの既存セッションをチェック
        existing_session_id = None
        existing_session = None

        # メモリ上のセッションをチェック
        for sid, session in _sessions.items():
            if Path(session.video_path).stem == video_id:
                logger.info(f"メモリ上に既存セッションを発見: {sid}")
                existing_session_id = sid
                existing_session = session
                break

        # データベースから既存セッションを探す
        if not existing_session_id:
            # プロジェクトルートからの絶対パスを使用
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            db_path = project_root / "web_interface" / "data" / "training" / "game_scene_labels.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()

                try:
                    # 同じビデオIDのセッションを全て取得
                    cursor.execute(
                        """
                        SELECT session_id, labeled_frames
                        FROM labeling_sessions
                        WHERE video_id = ?
                        ORDER BY labeled_frames DESC
                    """,
                        (video_id,),
                    )

                    sessions = cursor.fetchall()
                    if sessions:
                        # 最もラベル数が多いセッションを使用
                        existing_session_id = sessions[0][0]
                        logger.info(
                            f"データベースから既存セッション発見: {existing_session_id} (ラベル数: {sessions[0][1]})"
                        )

                        # 他のセッション（ラベル数が少ない）を削除
                        if len(sessions) > 1:
                            for sid, labeled_count in sessions[1:]:
                                logger.info(
                                    f"ラベル数の少ないセッションを削除: {sid} (ラベル数: {labeled_count})"
                                )
                                cursor.execute(
                                    """
                                    DELETE FROM labeling_sessions
                                    WHERE session_id = ?
                                """,
                                    (sid,),
                                )
                            conn.commit()

                except sqlite3.Error as e:
                    logger.error(f"データベースエラー: {e}")
                finally:
                    conn.close()

        # セッション作成または再開
        if existing_session_id:
            # 既存セッションを使用
            session_id = existing_session_id
            if existing_session:
                # メモリ上に既にある場合はそれを使用
                session = existing_session
                logger.info(f"既存のメモリ上のセッションを再利用: {session_id}")
            else:
                # データベースから読み込んで新しいセッションを作成
                classifier = SceneLabelingAPI.get_classifier()
                try:
                    session = SceneLabelingSession(
                        session_id=session_id, video_path=video_path, classifier=classifier
                    )
                    _sessions[session_id] = session
                    logger.info(f"データベースからセッションを復元: {session_id}")
                except Exception as e:
                    logger.error(f"SceneLabelingSession作成エラー: {e}")
                    logger.error(f"エラーの詳細: {type(e).__name__}: {str(e)}")
                    import traceback

                    logger.error(f"スタックトレース:\n{traceback.format_exc()}")
                    return jsonify({"error": f"セッション作成に失敗しました: {str(e)}"}), 500
        else:
            # 新規セッション作成
            if not session_id:
                session_id = str(uuid.uuid4())

            classifier = SceneLabelingAPI.get_classifier()
            try:
                session = SceneLabelingSession(
                    session_id=session_id, video_path=video_path, classifier=classifier
                )
                _sessions[session_id] = session
                logger.info(f"新規セッションを作成: {session_id}")
            except Exception as e:
                logger.error(f"SceneLabelingSession作成エラー: {e}")
                logger.error(f"エラーの詳細: {type(e).__name__}: {str(e)}")
                import traceback

                logger.error(f"スタックトレース:\n{traceback.format_exc()}")
                return jsonify({"error": f"セッション作成に失敗しました: {str(e)}"}), 500

        # 統計情報を取得
        statistics = session.get_statistics()
        logger.info(f"セッション統計情報: {statistics}")

        # セッション情報を返す
        return jsonify(
            {
                "session_id": session_id,
                "video_info": {
                    "path": video_path,
                    "total_frames": session.total_frames,
                    "fps": session.fps,
                    "width": session.width,
                    "height": session.height,
                    "duration": session.total_frames / session.fps if session.fps > 0 else 0,
                },
                "statistics": statistics,
                "is_resumed": statistics["labeled_frames"] > 0,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@scene_labeling_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """セッション情報を取得"""
    # アクティブセッションから探す
    if session_id in _sessions:
        session = _sessions[session_id]
        return jsonify(
            {
                "session_id": session_id,
                "video_info": {
                    "path": session.video_path,
                    "total_frames": session.total_frames,
                    "fps": session.fps,
                    "width": session.width,
                    "height": session.height,
                },
                "statistics": session.get_statistics(),
                "is_active": True,
            }
        )

    # データベースから探す
    import sqlite3
    from pathlib import Path

    db_path = "web_interface/data/training/game_scene_labels.db"
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT video_id, video_path, total_frames, labeled_frames
                FROM labeling_sessions
                WHERE session_id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            if row:
                video_id, video_path, total_frames, labeled_frames = row
                return jsonify(
                    {
                        "session_id": session_id,
                        "video_info": {"path": video_path, "total_frames": total_frames},
                        "statistics": {
                            "total_frames": total_frames,
                            "labeled_frames": labeled_frames,
                            "progress": labeled_frames / total_frames if total_frames > 0 else 0,
                        },
                        "is_active": False,
                    }
                )
        except sqlite3.Error as e:
            _logger.error(f"データベースエラー: {e}")
        finally:
            conn.close()

    return jsonify({"error": "セッションが見つかりません"}), 404


@scene_labeling_bp.route("/sessions/<session_id>/frame/<int:frame_number>", methods=["GET"])
def get_frame(session_id: str, frame_number: int):
    """フレーム画像を取得"""
    logger = _logger

    if session_id not in _sessions:
        logger.error(f"セッションが見つかりません: {session_id}")
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]

    try:
        # フレーム番号の妥当性チェック
        if frame_number < 0 or frame_number >= session.total_frames:
            logger.error(
                f"無効なフレーム番号: {frame_number} (総フレーム数: {session.total_frames})"
            )
            return jsonify({"error": f"無効なフレーム番号: {frame_number}"}), 400

        logger.info(f"フレーム取得リクエスト: session_id={session_id}, frame_number={frame_number}")
        logger.info(
            f"セッション情報: video_path={session.video_path}, total_frames={session.total_frames}"
        )

        frame = session.get_frame(frame_number)

        if frame is None:
            logger.error(f"フレーム {frame_number} の取得に失敗 (session_id={session_id})")
            logger.error(
                f"動画情報: path={session.video_path}, total_frames={session.total_frames}"
            )
            return jsonify({"error": f"フレーム {frame_number} を取得できません"}), 404

        # 自動推論結果を取得
        auto_result = None
        try:
            if session.classifier:
                is_game, confidence = session.classifier.classify_frame(frame)
                auto_result = {"is_game_scene": is_game, "confidence": confidence}
        except Exception as e:
            logger.warning(f"自動推論でエラー (フレーム {frame_number}): {e}")
            auto_result = None

        # 既存のラベルを確認
        existing_label = session.labels.get(frame_number)

        # 画像をBase64エンコード
        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if len(buffer) == 0:
                raise ValueError("画像エンコードでバッファが空")
            image_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            logger.error(f"画像エンコードエラー (フレーム {frame_number}): {e}")
            logger.error(f"フレーム形状: {frame.shape if frame is not None else 'None'}")
            return jsonify({"error": f"画像エンコードに失敗: {str(e)}"}), 500

        logger.debug(f"フレーム {frame_number} の取得完了 (画像サイズ: {len(image_base64)} bytes)")

        return jsonify(
            {
                "frame_number": frame_number,
                "image": f"data:image/jpeg;base64,{image_base64}",
                "label": {
                    "is_game_scene": existing_label.is_game_scene,
                    "confidence": existing_label.confidence,
                    "annotator": existing_label.annotator,
                }
                if existing_label
                else None,
                "auto_result": auto_result,
            }
        )

    except Exception as e:
        logger.error(f"フレーム取得でエラー (session: {session_id}, frame: {frame_number}): {e}")
        logger.error(f"エラー詳細: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(f"スタックトレース:\n{traceback.format_exc()}")
        return jsonify({"error": f"フレーム取得エラー: {str(e)}"}), 500


@scene_labeling_bp.route("/sessions/<session_id>/label", methods=["POST"])
def label_frame(session_id: str):
    """フレームにラベルを付与"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    data = request.get_json()
    frame_number = data.get("frame_number")
    is_game_scene = data.get("is_game_scene")
    annotator = data.get("annotator", "manual")

    if frame_number is None or is_game_scene is None:
        return jsonify({"error": "frame_numberとis_game_sceneが必要です"}), 400

    session = _sessions[session_id]
    label = session.label_frame(frame_number, is_game_scene, annotator)

    return jsonify(
        {
            "success": True,
            "label": {
                "frame_number": label.frame_number,
                "is_game_scene": label.is_game_scene,
                "annotator": label.annotator,
                "created_at": label.created_at,
            },
            "statistics": session.get_statistics(),
        }
    )


@scene_labeling_bp.route("/sessions/<session_id>/batch_label", methods=["POST"])
def batch_label_frames(session_id: str):
    """複数フレームに一括ラベル付与"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    data = request.get_json()
    start_frame = data.get("start_frame")
    end_frame = data.get("end_frame")
    is_game_scene = data.get("is_game_scene")
    annotator = data.get("annotator", "manual")

    if start_frame is None or end_frame is None or is_game_scene is None:
        return jsonify({"error": "必要なパラメータが不足しています"}), 400

    session = _sessions[session_id]
    labels = session.batch_label_frames(start_frame, end_frame, is_game_scene, annotator)

    return jsonify(
        {"success": True, "labeled_count": len(labels), "statistics": session.get_statistics()}
    )


@scene_labeling_bp.route("/sessions/<session_id>/next_unlabeled", methods=["GET"])
def get_next_unlabeled(session_id: str):
    """次の未ラベルフレームを取得"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    start_from = request.args.get("start_from", type=int)

    next_frame = session.get_next_unlabeled_frame(start_from)

    if next_frame is None:
        return jsonify({"next_frame": None, "message": "全てのフレームがラベル済みです"})

    return jsonify({"next_frame": next_frame})


@scene_labeling_bp.route("/sessions/<session_id>/uncertainty_frame", methods=["GET"])
def get_uncertainty_frame(session_id: str):
    """不確実性の高いフレームを取得"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    frame_number = session.get_uncertainty_frame()

    if frame_number is None:
        return jsonify({"frame_number": None, "message": "不確実なフレームはありません"})

    return jsonify({"frame_number": frame_number})


@scene_labeling_bp.route("/sessions/<session_id>/auto_label", methods=["POST"])
def auto_label(session_id: str):
    """自動ラベリング実行"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    data = request.get_json()
    frame_numbers = data.get("frame_numbers", [])
    sample_interval = data.get("sample_interval", 30)

    session = _sessions[session_id]

    if not frame_numbers:
        # 全フレームを対象にサンプリング
        frame_numbers = list(range(0, session.total_frames, sample_interval))

    # 自動ラベリング実行
    success_count = 0
    for frame_num in frame_numbers:
        if frame_num not in session.labels:  # 既存ラベルは上書きしない
            label = session.auto_label_frame(frame_num)
            if label:
                success_count += 1

    return jsonify(
        {
            "success": True,
            "labeled_count": success_count,
            "total_attempted": len(frame_numbers),
            "statistics": session.get_statistics(),
        }
    )


@scene_labeling_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """セッションを削除"""
    logger = _logger

    try:
        # メモリ上のセッションを削除
        deleted_from_memory = False
        if session_id in _sessions:
            del _sessions[session_id]
            deleted_from_memory = True
            logger.info(f"メモリ上のセッション {session_id} を削除しました")

        # データベースからも削除
        deleted_rows = 0
        db_path = "web_interface/data/training/game_scene_labels.db"
        if Path(db_path).exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            try:
                # セッション情報を削除
                cursor.execute(
                    """
                    DELETE FROM labeling_sessions
                    WHERE session_id = ?
                """,
                    (session_id,),
                )

                # セッションに関連するラベルも削除（必要に応じて）
                # cursor.execute("""
                #     DELETE FROM game_scene_labels
                #     WHERE video_id IN (
                #         SELECT video_id FROM labeling_sessions WHERE session_id = ?
                #     )
                # """, (session_id,))

                conn.commit()
                deleted_rows = cursor.rowcount
                logger.info(f"データベースから {deleted_rows} 行を削除しました")

            except sqlite3.Error as e:
                logger.error(f"データベース削除エラー: {e}")
                conn.rollback()
                return jsonify({"error": f"データベース削除エラー: {str(e)}"}), 500
            finally:
                conn.close()

        if deleted_from_memory or deleted_rows > 0:
            return jsonify(
                {
                    "success": True,
                    "deleted_from_memory": deleted_from_memory,
                    "deleted_from_db": deleted_rows > 0,
                }
            )
        else:
            return jsonify({"error": "セッションが見つかりません"}), 404

    except Exception as e:
        logger.error(f"セッション削除エラー: {e}")
        return jsonify({"error": str(e)}), 500


@scene_labeling_bp.route("/sessions/<session_id>/segments", methods=["GET"])
def get_segments(session_id: str):
    """セグメント情報を取得"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    segments = session.export_segments()

    return jsonify(
        {
            "segments": segments,
            "total_segments": len(segments),
            "game_segments": sum(1 for s in segments if s["scene_type"] == "game"),
            "non_game_segments": sum(1 for s in segments if s["scene_type"] == "non_game"),
        }
    )


@scene_labeling_bp.route("/sessions/<session_id>/close", methods=["POST"])
def close_session(session_id: str):
    """セッションを終了"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    statistics = session.get_statistics()

    # セグメントをエクスポート
    segments = session.export_segments()

    # セッションを閉じる
    session.close()
    del _sessions[session_id]

    return jsonify(
        {"success": True, "final_statistics": statistics, "exported_segments": len(segments)}
    )


@scene_labeling_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """セッション一覧を取得（アクティブ＋保存済み）"""
    import sqlite3
    from pathlib import Path

    _logger.info("list_sessions() が呼び出されました")

    # ビデオIDごとに最新のセッションのみを保持する辞書
    video_sessions = {}

    # アクティブなセッションを処理
    _logger.info(f"アクティブセッション数: {len(_sessions)}")
    for session_id, session in _sessions.items():
        video_id = Path(session.video_path).stem
        session_info = {
            "session_id": session_id,
            "video_id": video_id,
            "video_path": session.video_path,
            "statistics": session.get_statistics(),
            "is_active": True,
            "created_at": None,
            "updated_at": None,
        }

        # 同じビデオIDのセッションがない、またはよりラベル数が多い場合に更新
        if (
            video_id not in video_sessions
            or session_info["statistics"]["labeled_frames"]
            > video_sessions[video_id]["statistics"]["labeled_frames"]
        ):
            video_sessions[video_id] = session_info

    # データベースから保存済みセッションを取得
    # プロジェクトルートからの絶対パスを使用
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    db_path = project_root / "web_interface" / "data" / "training" / "game_scene_labels.db"
    _logger.info(f"データベースパス: {db_path}, 存在: {db_path.exists()}")
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT session_id, video_id, video_path, total_frames, labeled_frames,
                       created_at, updated_at
                FROM labeling_sessions
                ORDER BY labeled_frames DESC, updated_at DESC
            """)

            rows = cursor.fetchall()
            _logger.info(f"データベースから取得したセッション数: {len(rows)}")

            for row in rows:
                (
                    session_id,
                    video_id,
                    video_path,
                    total_frames,
                    labeled_frames,
                    created_at,
                    updated_at,
                ) = row

                # アクティブセッションと重複する場合はスキップ
                if session_id not in _sessions:
                    _logger.debug(
                        f"データベースセッション: {session_id}, video_id={video_id}, labeled={labeled_frames}"
                    )
                    session_info = {
                        "session_id": session_id,
                        "video_id": video_id,
                        "video_path": video_path,
                        "statistics": {
                            "total_frames": total_frames,
                            "labeled_frames": labeled_frames,
                            "progress": labeled_frames / total_frames if total_frames > 0 else 0,
                        },
                        "is_active": False,
                        "created_at": created_at,
                        "updated_at": updated_at,
                    }

                    # 同じビデオIDのセッションがない、またはよりラベル数が多い場合に更新
                    if (
                        video_id not in video_sessions
                        or labeled_frames > video_sessions[video_id]["statistics"]["labeled_frames"]
                    ):
                        video_sessions[video_id] = session_info

        except sqlite3.Error as e:
            _logger.error(f"データベースエラー: {e}")
        finally:
            conn.close()

    # 辞書から値のリストに変換
    sessions_info = list(video_sessions.values())
    _logger.info(
        f"最終的なセッション数: {len(sessions_info)}, ビデオID別: {list(video_sessions.keys())}"
    )

    # 更新日時でソート（新しい順）
    sessions_info.sort(key=lambda x: x.get("updated_at") or "", reverse=True)

    return jsonify({"sessions": sessions_info, "total": len(sessions_info)})
