"""APIルート定義"""

from flask import Blueprint

from .auto_label_routes import auto_label_bp
from .frame_routes import frame_bp
from .labeling_routes import labeling_bp
from .session_routes import session_bp


def create_scene_labeling_blueprint() -> Blueprint:
    """
    対局画面ラベリングAPIのBlueprintを作成

    Returns:
        統合されたBlueprint
    """
    # メインのBlueprint
    scene_labeling_bp = Blueprint("scene_labeling", __name__, url_prefix="/api/scene_labeling")

    # サブBlueprintを登録
    scene_labeling_bp.register_blueprint(session_bp)
    scene_labeling_bp.register_blueprint(frame_bp)
    scene_labeling_bp.register_blueprint(labeling_bp)
    scene_labeling_bp.register_blueprint(auto_label_bp)

    return scene_labeling_bp


__all__ = ["create_scene_labeling_blueprint"]
