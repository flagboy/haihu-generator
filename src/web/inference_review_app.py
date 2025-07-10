"""
推測フレームレビュー用Webアプリケーション

推測されたアクションのフレームを確認し、修正できるWebインターフェース。
"""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

from ..tracking.inference_frame_manager import InferenceFrameManager
from ..utils.logger import LoggerMixin


class InferenceReviewApp(LoggerMixin):
    """推測フレームレビュー用アプリケーション"""

    def __init__(self, base_dir: str = "inference_frames"):
        """
        初期化

        Args:
            base_dir: フレーム保存用のベースディレクトリ
        """
        self.app = Flask(__name__)
        self.base_dir = Path(base_dir)
        self.frame_manager = None
        self._setup_routes()

    def _setup_routes(self):
        """ルートを設定"""

        @self.app.route("/")
        def index():
            """セッション一覧を表示"""
            sessions = []
            if self.base_dir.exists():
                for session_dir in sorted(self.base_dir.iterdir(), reverse=True):
                    if session_dir.is_dir():
                        index_file = session_dir / "index.json"
                        if index_file.exists():
                            with open(index_file, encoding="utf-8") as f:
                                data = json.load(f)
                                sessions.append(
                                    {
                                        "id": session_dir.name,
                                        "frame_count": len(data),
                                        "path": str(session_dir),
                                    }
                                )

            html = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>推測フレームレビュー</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .session-list { list-style: none; padding: 0; }
        .session-item {
            border: 1px solid #ccc;
            margin: 10px 0;
            padding: 15px;
            cursor: pointer;
        }
        .session-item:hover { background-color: #f0f0f0; }
    </style>
</head>
<body>
    <h1>推測フレームレビュー - セッション一覧</h1>
    <ul class="session-list">
        {% for session in sessions %}
        <li class="session-item" onclick="window.location.href='/session/{{ session.id }}'">
            <h3>{{ session.id }}</h3>
            <p>フレーム数: {{ session.frame_count }}</p>
        </li>
        {% endfor %}
    </ul>
</body>
</html>
"""
            from flask import render_template_string

            return render_template_string(html, sessions=sessions)

        @self.app.route("/session/<session_id>")
        def session_view(session_id):
            """セッションのフレーム一覧を表示"""
            self.frame_manager = InferenceFrameManager(str(self.base_dir))
            self.frame_manager.load_session(session_id)

            frames = self.frame_manager.frames_index
            unverified_count = len(self.frame_manager.get_unverified_frames())

            html = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>推測フレームレビュー - {{ session_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .nav { margin-bottom: 20px; }
        .frame-container {
            border: 1px solid #ccc;
            margin: 20px 0;
            padding: 15px;
            display: none;
        }
        .frame-container.active { display: block; }
        .frame-image { max-width: 100%; margin: 10px 0; }
        .frame-info { margin: 10px 0; background: #f5f5f5; padding: 10px; }
        .correction-form { margin: 15px 0; padding: 10px; background: #f0f0f0; }
        .verified { border-color: #28a745; background-color: #d4edda; }
        .unverified { border-color: #dc3545; background-color: #f8d7da; }
        button { margin: 5px; padding: 8px 15px; cursor: pointer; }
        .btn-primary { background: #007bff; color: white; border: none; }
        .btn-success { background: #28a745; color: white; border: none; }
        .btn-danger { background: #dc3545; color: white; border: none; }
        .navigation { margin: 20px 0; text-align: center; }
        .stats { background: #e9ecef; padding: 15px; margin: 20px 0; }
        input, select, textarea {
            padding: 5px;
            margin: 5px 0;
            width: 100%;
            box-sizing: border-box;
        }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">← セッション一覧に戻る</a>
    </div>

    <h1>推測フレームレビュー - セッション: {{ session_id }}</h1>

    <div class="stats">
        <p>合計フレーム数: {{ total_frames }}</p>
        <p>未検証: {{ unverified_count }}</p>
        <p>検証済み: {{ total_frames - unverified_count }}</p>
    </div>

    <div class="navigation">
        <button onclick="previousFrame()" class="btn-primary">← 前へ</button>
        <span id="frame-counter">1 / {{ total_frames }}</span>
        <button onclick="nextFrame()" class="btn-primary">次へ →</button>
        <button onclick="nextUnverified()" class="btn-danger">次の未検証へ →</button>
    </div>

    {% for i, frame in enumerate(frames) %}
    <div class="frame-container {% if frame.human_verified %}verified{% else %}unverified{% endif %}"
         id="frame-{{ i }}" data-frame-id="{{ frame.frame_id }}">
        <h3>フレーム: {{ frame.frame_id }}</h3>
        <img src="/image/{{ session_id }}/{{ frame.frame_id }}.jpg" class="frame-image">

        <div class="frame-info">
            <table style="width: 100%;">
                <tr>
                    <td><strong>フレーム番号:</strong> {{ frame.frame_number }}</td>
                    <td><strong>巡番号:</strong> {{ frame.turn_number }}</td>
                    <td><strong>プレイヤー:</strong> {{ frame.player_index }}</td>
                </tr>
                <tr>
                    <td><strong>推測アクション:</strong> {{ frame.action_type }}</td>
                    <td><strong>推測牌:</strong> {{ frame.inferred_tile or "なし" }}</td>
                    <td><strong>信頼度:</strong> {{ "%.2f"|format(frame.confidence) }}</td>
                </tr>
            </table>
            <p><strong>理由:</strong> {{ frame.reason }}</p>
            <p><strong>前巡の手牌:</strong> {{ ' '.join(frame.prev_hand) }}</p>
            <p><strong>現在の手牌:</strong> {{ ' '.join(frame.curr_hand) }}</p>
        </div>

        <div class="correction-form">
            <h4>修正フォーム</h4>
            <form id="form-{{ i }}">
                <div>
                    <label>アクションタイプ:</label>
                    <select name="action_type" id="action-type-{{ i }}">
                        <option value="">変更なし</option>
                        <option value="draw">ツモ</option>
                        <option value="discard">捨て牌</option>
                        <option value="pon">ポン</option>
                        <option value="chi">チー</option>
                        <option value="kan">カン</option>
                        <option value="reach">リーチ</option>
                        <option value="none">アクションなし</option>
                    </select>
                </div>
                <div>
                    <label>牌:</label>
                    <input type="text" name="tile" id="tile-{{ i }}"
                           placeholder="例: 1m, 5p, 7s"
                           value="{{ frame.human_correction.tile if frame.human_correction else '' }}">
                </div>
                <div>
                    <label>コメント:</label>
                    <textarea name="comment" id="comment-{{ i }}" rows="3">{{ frame.human_correction.comment if frame.human_correction else '' }}</textarea>
                </div>
                <button type="button" onclick="saveCorrection({{ i }})" class="btn-primary">保存</button>
                <button type="button" onclick="markVerified({{ i }})" class="btn-success">検証済みにする</button>
            </form>
        </div>
    </div>
    {% endfor %}

    <script>
        let currentFrame = 0;
        const totalFrames = {{ total_frames }};
        const frames = document.querySelectorAll('.frame-container');

        function showFrame(index) {
            frames.forEach((f, i) => {
                f.classList.toggle('active', i === index);
            });
            document.getElementById('frame-counter').textContent = `${index + 1} / ${totalFrames}`;
            currentFrame = index;
        }

        function previousFrame() {
            if (currentFrame > 0) {
                showFrame(currentFrame - 1);
            }
        }

        function nextFrame() {
            if (currentFrame < totalFrames - 1) {
                showFrame(currentFrame + 1);
            }
        }

        function nextUnverified() {
            for (let i = currentFrame + 1; i < totalFrames; i++) {
                if (frames[i].classList.contains('unverified')) {
                    showFrame(i);
                    return;
                }
            }
            // 最初から探す
            for (let i = 0; i <= currentFrame; i++) {
                if (frames[i].classList.contains('unverified')) {
                    showFrame(i);
                    return;
                }
            }
        }

        async function saveCorrection(index) {
            const frameId = frames[index].dataset.frameId;
            const actionType = document.getElementById(`action-type-${index}`).value;
            const tile = document.getElementById(`tile-${index}`).value;
            const comment = document.getElementById(`comment-${index}`).value;

            const correction = {};
            if (actionType) correction.action_type = actionType;
            if (tile) correction.tile = tile;
            if (comment) correction.comment = comment;

            const response = await fetch(`/api/correction/${frameId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({correction, verified: false})
            });

            if (response.ok) {
                alert('修正を保存しました');
            } else {
                alert('保存に失敗しました');
            }
        }

        async function markVerified(index) {
            const frameId = frames[index].dataset.frameId;

            const response = await fetch(`/api/verify/${frameId}`, {
                method: 'POST'
            });

            if (response.ok) {
                frames[index].classList.remove('unverified');
                frames[index].classList.add('verified');
                alert('検証済みにしました');
            } else {
                alert('失敗しました');
            }
        }

        // 初期表示
        showFrame(0);

        // キーボードショートカット
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') previousFrame();
            else if (e.key === 'ArrowRight') nextFrame();
            else if (e.key === 'u') nextUnverified();
            else if (e.key === 'v') markVerified(currentFrame);
        });
    </script>
</body>
</html>
"""
            return render_template_string(
                html,
                session_id=session_id,
                frames=frames,
                total_frames=len(frames),
                unverified_count=unverified_count,
                enumerate=enumerate,
            )

        @self.app.route("/image/<session_id>/<filename>")
        def serve_image(session_id, filename):
            """画像を提供"""
            image_path = self.base_dir / session_id / "images" / filename
            if image_path.exists():
                return send_file(str(image_path))
            return "Image not found", 404

        @self.app.route("/api/correction/<frame_id>", methods=["POST"])
        def save_correction(frame_id):
            """修正を保存"""
            data = request.json
            correction = data.get("correction", {})
            verified = data.get("verified", False)

            if self.frame_manager:
                try:
                    self.frame_manager.update_frame_correction(frame_id, correction, verified)
                    return jsonify({"status": "success"})
                except Exception as e:
                    return jsonify({"status": "error", "message": str(e)}), 500

            return jsonify({"status": "error", "message": "No session loaded"}), 400

        @self.app.route("/api/verify/<frame_id>", methods=["POST"])
        def mark_verified(frame_id):
            """検証済みにする"""
            if self.frame_manager:
                try:
                    self.frame_manager.update_frame_correction(frame_id, {}, True)
                    return jsonify({"status": "success"})
                except Exception as e:
                    return jsonify({"status": "error", "message": str(e)}), 500

            return jsonify({"status": "error", "message": "No session loaded"}), 400

        @self.app.route("/api/export/<session_id>")
        def export_corrections(session_id):
            """修正情報をエクスポート"""
            manager = InferenceFrameManager(str(self.base_dir))
            manager.load_session(session_id)
            corrections = manager.export_corrections()
            return jsonify(corrections)

    def run(self, host: str = "0.0.0.0", port: int = 5001, debug: bool = True):
        """アプリケーションを起動"""
        self.logger.info(f"推測フレームレビューアプリを起動: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app = InferenceReviewApp()
    app.run()
