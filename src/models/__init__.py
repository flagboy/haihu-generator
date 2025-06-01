"""
モデル管理モジュール
"""

# 重い依存関係の遅延読み込み
ModelManager = None
TenhouAction = None
TenhouActionType = None
TenhouAgariAction = None
TenhouCallAction = None
TenhouCallType = None
TenhouDiscardAction = None
TenhouDrawAction = None
TenhouGameData = None
TenhouGameDataBuilder = None
TenhouGameResult = None
TenhouGameRule = None
TenhouGameType = None
TenhouPlayerState = None
TenhouRiichiAction = None
TenhouTile = None


def _lazy_import():
    """重い依存関係の遅延読み込み"""
    global ModelManager, TenhouAction, TenhouActionType, TenhouAgariAction
    global TenhouCallAction, TenhouCallType, TenhouDiscardAction, TenhouDrawAction
    global TenhouGameData, TenhouGameDataBuilder, TenhouGameResult, TenhouGameRule
    global TenhouGameType, TenhouPlayerState, TenhouRiichiAction, TenhouTile

    try:
        from .model_manager import ModelManager
        from .tenhou_game_data import (
            TenhouAction,
            TenhouActionType,
            TenhouAgariAction,
            TenhouCallAction,
            TenhouCallType,
            TenhouDiscardAction,
            TenhouDrawAction,
            TenhouGameData,
            TenhouGameDataBuilder,
            TenhouGameResult,
            TenhouGameRule,
            TenhouGameType,
            TenhouPlayerState,
            TenhouRiichiAction,
            TenhouTile,
        )
    except ImportError:
        # 依存関係が不足している場合はスキップ
        pass


__all__ = [
    "ModelManager",
    "TenhouGameData",
    "TenhouAction",
    "TenhouTile",
    "TenhouPlayerState",
    "TenhouGameRule",
    "TenhouGameResult",
    "TenhouGameDataBuilder",
    "TenhouActionType",
    "TenhouCallType",
    "TenhouGameType",
    "TenhouDrawAction",
    "TenhouDiscardAction",
    "TenhouCallAction",
    "TenhouRiichiAction",
    "TenhouAgariAction",
]
