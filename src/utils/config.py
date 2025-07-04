"""
設定管理モジュール
"""

from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


class ConfigManager:
    """設定ファイルの読み込みと管理を行うクラス"""

    def __init__(self, config_path: str | Path | None = None):
        """
        設定管理クラスの初期化

        Args:
            config_path: 設定ファイルのパス（デフォルト: config.yaml）
        """
        if config_path is None:
            # プロジェクトルートのconfig.yamlを使用
            project_root = Path(__file__).parent.parent.parent
            self.config_path = project_root / "config.yaml"
        else:
            self.config_path = Path(config_path)
        self._config = self._load_config()
        self._ensure_directories()

    def _load_config(self) -> dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return cast(dict[str, Any], config)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}") from err
        except yaml.YAMLError as e:
            raise ValueError(f"設定ファイルの形式が正しくありません: {e}") from e

    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        directories = self.get("directories", {})
        for _dir_name, dir_path in directories.items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得

        Args:
            key: 設定キー（ドット記法対応: "video.frame_extraction.fps"）
            default: デフォルト値

        Returns:
            設定値
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_config(self) -> dict[str, Any]:
        """
        全設定を取得

        Returns:
            設定辞書
        """
        return self._config.copy()

    def get_image_config(self) -> dict[str, Any]:
        """画像処理設定を取得"""
        return cast(dict[str, Any], self.get("image", {}))

    def get_system_config(self) -> dict[str, Any]:
        """システム設定を取得"""
        return cast(dict[str, Any], self.get("system", {}))

    def update_config(self, key: str, value: Any):
        """
        設定値を更新

        Args:
            key: 設定キー（ドット記法対応）
            value: 新しい値
        """
        keys = key.split(".")
        config = self._config

        # 最後のキー以外をたどる
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 最後のキーに値を設定
        config[keys[-1]] = value

    def save_config(self):
        """設定をファイルに保存"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self._config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )

    # ====================
    # 天鳳形式対応メソッド
    # ====================

    def get_tenhou_config(self) -> dict[str, Any]:
        """天鳳形式設定を取得"""
        return cast(dict[str, Any], self.get("tenhou", {}))

    def get_tenhou_output_config(self) -> dict[str, Any]:
        """天鳳出力設定を取得"""
        return cast(dict[str, Any], self.get("tenhou.output", {}))

    def get_tenhou_notation(self) -> dict[str, Any]:
        """天鳳牌表記を取得"""
        return cast(dict[str, Any], self.get("tenhou.specification.tile_notation", {}))

    def get_ai_config(self) -> dict[str, Any]:
        """AI/ML設定を取得"""
        return cast(dict[str, Any], self.get("ai", {}))

    def get_pipeline_config(self) -> dict[str, Any]:
        """パイプライン設定を取得"""
        return cast(dict[str, Any], self.get("pipeline", {}))

    def get_optimization_config(self) -> dict[str, Any]:
        """最適化設定を取得"""
        return cast(dict[str, Any], self.get("optimization", {}))

    # ====================
    # 後方互換性メソッド
    # ====================

    def get_directories(self) -> dict[str, str]:
        """ディレクトリ設定を取得（新旧両形式対応）"""
        # 新形式チェック
        if "system" in self._config and "output" in self._config["system"]:
            output_dir = self.get("system.output.directory", "data/output")
            return {
                "input": "data/input",
                "output": output_dir,
                "temp": "data/temp",
                "models": "models",
                "logs": "logs",
            }

        # 旧形式フォールバック
        return cast(dict[str, str], self.get("directories", {}))

    def get_video_config(self) -> dict[str, Any]:
        """動画処理設定を取得（新旧両形式対応）"""
        # 新形式チェック
        if "pipeline" in self._config and "video" in self._config["pipeline"]:
            return cast(dict[str, Any], self.get("pipeline.video", {}))

        # 旧形式フォールバック
        return cast(dict[str, Any], self.get("video", {}))

    def get_logging_config(self) -> dict[str, Any]:
        """ログ設定を取得（新旧両形式対応）"""
        # 新形式チェック
        if "system" in self._config and "logging" in self._config["system"]:
            return cast(dict[str, Any], self.get("system.logging", {}))

        # 旧形式フォールバック
        return cast(dict[str, Any], self.get("logging", {}))

    def get_tile_definitions(self) -> dict[str, Any]:
        """麻雀牌定義を取得（新旧両形式対応）"""
        # 新形式チェック
        if "tenhou" in self._config:
            notation = self.get("tenhou.specification.tile_notation", {})
            if notation:
                return cast(dict[str, Any], notation)

        # 旧形式フォールバック
        return cast(dict[str, Any], self.get("tiles", {}))
