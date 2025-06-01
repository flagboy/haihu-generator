"""
設定管理モジュールのテスト
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.utils.config import ConfigManager


class TestConfigManager:
    """ConfigManagerクラスのテスト"""
    
    def test_load_default_config(self):
        """デフォルト設定ファイルの読み込みテスト"""
        config_manager = ConfigManager()
        
        # 基本的な設定項目が存在することを確認
        assert config_manager.get("video") is not None
        assert config_manager.get("image") is not None
        assert config_manager.get("logging") is not None
        assert config_manager.get("directories") is not None
        assert config_manager.get("tiles") is not None
        assert config_manager.get("system") is not None
    
    def test_get_with_dot_notation(self):
        """ドット記法での設定取得テスト"""
        config_manager = ConfigManager()
        
        # ドット記法で設定を取得
        fps = config_manager.get("video.frame_extraction.fps")
        assert fps is not None
        
        # 存在しないキーの場合はデフォルト値を返す
        non_existent = config_manager.get("non.existent.key", "default")
        assert non_existent == "default"
    
    def test_get_specific_configs(self):
        """特定設定取得メソッドのテスト"""
        config_manager = ConfigManager()
        
        video_config = config_manager.get_video_config()
        assert isinstance(video_config, dict)
        
        image_config = config_manager.get_image_config()
        assert isinstance(image_config, dict)
        
        logging_config = config_manager.get_logging_config()
        assert isinstance(logging_config, dict)
        
        directories = config_manager.get_directories()
        assert isinstance(directories, dict)
        
        tile_definitions = config_manager.get_tile_definitions()
        assert isinstance(tile_definitions, dict)
        
        system_config = config_manager.get_system_config()
        assert isinstance(system_config, dict)
    
    def test_update_config(self):
        """設定更新のテスト"""
        # 一時的な設定ファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                "test": {
                    "value": 123,
                    "nested": {
                        "item": "original"
                    }
                }
            }
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            config_manager = ConfigManager(temp_config_path)
            
            # 設定を更新
            config_manager.update_config("test.value", 456)
            config_manager.update_config("test.nested.item", "updated")
            config_manager.update_config("test.new_key", "new_value")
            
            # 更新された値を確認
            assert config_manager.get("test.value") == 456
            assert config_manager.get("test.nested.item") == "updated"
            assert config_manager.get("test.new_key") == "new_value"
            
        finally:
            Path(temp_config_path).unlink()
    
    def test_custom_config_file(self):
        """カスタム設定ファイルのテスト"""
        # 一時的な設定ファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                "custom": {
                    "setting": "test_value"
                }
            }
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            config_manager = ConfigManager(temp_config_path)
            assert config_manager.get("custom.setting") == "test_value"
            
        finally:
            Path(temp_config_path).unlink()
    
    def test_file_not_found(self):
        """存在しない設定ファイルのテスト"""
        with pytest.raises(FileNotFoundError):
            ConfigManager("non_existent_config.yaml")
    
    def test_invalid_yaml(self):
        """不正なYAMLファイルのテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_config_path = f.name
        
        try:
            with pytest.raises(ValueError):
                ConfigManager(temp_config_path)
                
        finally:
            Path(temp_config_path).unlink()