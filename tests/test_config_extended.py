"""
ConfigManager の拡張テスト - カバレッジ向上のため
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import ConfigManager


class TestConfigManagerExtended:
    """ConfigManagerの拡張テストクラス"""

    @pytest.fixture
    def temp_config_file(self):
        """一時的な設定ファイルを作成するフィクスチャ"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "video": {"frame_extraction": {"fps": 30, "format": "jpg"}},
                "image": {"size": [640, 480], "quality": 95},
                "system": {
                    "output": {"directory": "test_output"},
                    "logging": {"level": "DEBUG", "file": "test.log"},
                },
                "tenhou": {
                    "output": {"format": "json"},
                    "specification": {
                        "tile_notation": {
                            "manzu": ["1m", "2m", "3m"],
                            "pinzu": ["1p", "2p", "3p"],
                            "souzu": ["1s", "2s", "3s"],
                        }
                    },
                },
                "ai": {"detection": {"confidence": 0.8}},
                "pipeline": {"video": {"preprocessing": True}, "batch_size": 32},
                "optimization": {"gpu": True, "memory_limit": "8GB"},
                "tiles": {"types": ["manzu", "pinzu", "souzu", "jihai"]},
                "logging": {"level": "INFO"},
                "directories": {"input": "old_input", "output": "old_output"},
            }
            yaml.dump(config_data, f)
            yield f.name
        Path(f.name).unlink()

    def test_get_config_full(self, temp_config_file):
        """全設定取得のテスト"""
        config_manager = ConfigManager(temp_config_file)
        full_config = config_manager.get_config()

        assert isinstance(full_config, dict)
        assert "video" in full_config
        assert "system" in full_config
        assert full_config is not config_manager._config  # 異なるオブジェクトであることを確認

        # コピーであることを確認（値を変更してもオリジナルに影響しない）
        full_config["test_key"] = "test_value"
        assert "test_key" not in config_manager._config

    def test_save_config(self, temp_config_file):
        """設定保存のテスト"""
        config_manager = ConfigManager(temp_config_file)

        # 設定を更新
        config_manager.update_config("test.save", "saved_value")
        config_manager.save_config()

        # 新しいインスタンスで読み込み直して確認
        new_config_manager = ConfigManager(temp_config_file)
        assert new_config_manager.get("test.save") == "saved_value"

    def test_tenhou_config_methods(self, temp_config_file):
        """天鳳関連設定メソッドのテスト"""
        config_manager = ConfigManager(temp_config_file)

        # get_tenhou_config
        tenhou_config = config_manager.get_tenhou_config()
        assert isinstance(tenhou_config, dict)
        assert "output" in tenhou_config

        # get_tenhou_output_config
        output_config = config_manager.get_tenhou_output_config()
        assert output_config["format"] == "json"

        # get_tenhou_notation
        notation = config_manager.get_tenhou_notation()
        assert "manzu" in notation
        assert notation["manzu"] == ["1m", "2m", "3m"]

    def test_ai_and_pipeline_config(self, temp_config_file):
        """AI・パイプライン設定メソッドのテスト"""
        config_manager = ConfigManager(temp_config_file)

        # get_ai_config
        ai_config = config_manager.get_ai_config()
        assert ai_config["detection"]["confidence"] == 0.8

        # get_pipeline_config
        pipeline_config = config_manager.get_pipeline_config()
        assert pipeline_config["batch_size"] == 32

        # get_optimization_config
        opt_config = config_manager.get_optimization_config()
        assert opt_config["gpu"] is True
        assert opt_config["memory_limit"] == "8GB"

    def test_backward_compatibility_methods(self, temp_config_file):
        """後方互換性メソッドのテスト"""
        config_manager = ConfigManager(temp_config_file)

        # get_directories - 新形式
        directories = config_manager.get_directories()
        assert directories["output"] == "test_output"  # system.output.directory から取得

        # get_video_config - 新形式
        video_config = config_manager.get_video_config()
        assert video_config["preprocessing"] is True  # pipeline.video から取得

        # get_logging_config - 新形式
        logging_config = config_manager.get_logging_config()
        assert logging_config["level"] == "DEBUG"  # system.logging から取得

        # get_tile_definitions - 新形式
        tiles = config_manager.get_tile_definitions()
        assert "manzu" in tiles  # tenhou.specification.tile_notation から取得

    def test_backward_compatibility_fallback(self):
        """後方互換性フォールバックのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 旧形式の設定ファイル
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                old_config = {
                    "video": {"old_format": True},
                    "logging": {"old_level": "WARN"},
                    "tiles": {"old_tiles": ["1m", "2m"]},
                    "directories": {"old_dir": f"{temp_dir}/old_path"},
                }
                yaml.dump(old_config, f)
                temp_path = f.name

            try:
                config_manager = ConfigManager(temp_path)

                # 旧形式フォールバック確認
                assert config_manager.get_video_config()["old_format"] is True
                assert config_manager.get_logging_config()["old_level"] == "WARN"
                assert config_manager.get_tile_definitions()["old_tiles"] == ["1m", "2m"]
                assert config_manager.get_directories()["old_dir"] == f"{temp_dir}/old_path"

            finally:
                Path(temp_path).unlink()

    def test_get_with_nested_default(self, temp_config_file):
        """ネストされたデフォルト値のテスト"""
        config_manager = ConfigManager(temp_config_file)

        # 存在しないキーでTypeErrorが発生するケース
        result = config_manager.get("video.non_existent.deep.key", "default")
        assert result == "default"

    def test_update_config_deep_nesting(self, temp_config_file):
        """深いネストの設定更新テスト"""
        config_manager = ConfigManager(temp_config_file)

        # 深いネストの新規作成
        config_manager.update_config("new.deep.nested.key", "deep_value")
        assert config_manager.get("new.deep.nested.key") == "deep_value"

        # 既存の深いネストの更新
        config_manager.update_config("system.output.directory", "new_directory")
        assert config_manager.get("system.output.directory") == "new_directory"

    def test_ensure_directories_creation(self):
        """ディレクトリ作成のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                config_data = {
                    "directories": {"test1": f"{temp_dir}/test1/sub", "test2": f"{temp_dir}/test2"}
                }
                yaml.dump(config_data, f)
                temp_config = f.name

            try:
                # ConfigManager初期化時にディレクトリが作成される
                ConfigManager(temp_config)

                # ディレクトリが作成されたことを確認
                assert Path(f"{temp_dir}/test1/sub").exists()
                assert Path(f"{temp_dir}/test2").exists()

            finally:
                Path(temp_config).unlink()

    def test_empty_config_sections(self, temp_config_file):
        """空の設定セクションのテスト"""
        config_manager = ConfigManager(temp_config_file)

        # 存在しないセクションは空の辞書を返す
        assert config_manager.get("nonexistent_section", {}) == {}
        assert config_manager.get_tenhou_config().get("nonexistent", {}) == {}

    def test_yaml_safe_load(self):
        """YAMLの安全な読み込みテスト"""
        # yaml.safe_loadが危険なタグを拒否することを確認
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test: !!python/object/apply:os.system ['echo hacked']")
            temp_path = f.name

        try:
            # yaml.safe_load は危険なタグでValueErrorを発生させる
            with pytest.raises(ValueError, match="設定ファイルの形式が正しくありません"):
                ConfigManager(temp_path)
        finally:
            Path(temp_path).unlink()
