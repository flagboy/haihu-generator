"""
ファイル入出力のための共通ユーティリティ
"""

import json
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FileIOHelper:
    """ファイル入出力の共通処理を提供するヘルパークラス"""
    
    @staticmethod
    def save_json(data: Dict[str, Any], 
                  path: Union[str, Path], 
                  pretty: bool = True,
                  ensure_ascii: bool = False) -> None:
        """
        JSONファイルを保存
        
        Args:
            data: 保存するデータ
            path: 保存先のパス
            pretty: 整形して保存するか
            ensure_ascii: ASCII文字のみを使用するか
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        json_kwargs = {
            'ensure_ascii': ensure_ascii,
            'separators': (',', ':') if not pretty else (',', ': '),
        }
        if pretty:
            json_kwargs['indent'] = 2
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, **json_kwargs)
            logger.debug(f"JSON saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save JSON to {path}: {e}")
            raise
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict[str, Any]:
        """
        JSONファイルを読み込み
        
        Args:
            path: 読み込むファイルのパス
            
        Returns:
            読み込んだデータ
        """
        path = Path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"JSON loaded from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON from {path}: {e}")
            raise
    
    @staticmethod
    def save_yaml(data: Dict[str, Any],
                  path: Union[str, Path],
                  default_flow_style: bool = False) -> None:
        """
        YAMLファイルを保存
        
        Args:
            data: 保存するデータ
            path: 保存先のパス
            default_flow_style: フロースタイルを使用するか
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, 
                         default_flow_style=default_flow_style,
                         allow_unicode=True,
                         sort_keys=False)
            logger.debug(f"YAML saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save YAML to {path}: {e}")
            raise
    
    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """
        YAMLファイルを読み込み
        
        Args:
            path: 読み込むファイルのパス
            
        Returns:
            読み込んだデータ
        """
        path = Path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            logger.debug(f"YAML loaded from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load YAML from {path}: {e}")
            raise
    
    @staticmethod
    def save_pickle(data: Any,
                    path: Union[str, Path]) -> None:
        """
        Pickleファイルを保存
        
        Args:
            data: 保存するデータ
            path: 保存先のパス
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Pickle saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save pickle to {path}: {e}")
            raise
    
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Any:
        """
        Pickleファイルを読み込み
        
        Args:
            path: 読み込むファイルのパス
            
        Returns:
            読み込んだデータ
        """
        path = Path(path)
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Pickle loaded from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load pickle from {path}: {e}")
            raise
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        ディレクトリが存在することを保証
        
        Args:
            path: ディレクトリパス
            
        Returns:
            作成されたPathオブジェクト
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def safe_write(path: Union[str, Path],
                   content: Union[str, bytes],
                   mode: str = 'w',
                   encoding: Optional[str] = 'utf-8') -> None:
        """
        安全なファイル書き込み（一時ファイル経由）
        
        Args:
            path: 書き込み先パス
            content: 書き込む内容
            mode: ファイルモード
            encoding: エンコーディング（テキストモードの場合）
        """
        path = Path(path)
        temp_path = path.with_suffix(path.suffix + '.tmp')
        
        try:
            # 一時ファイルに書き込み
            if 'b' in mode:
                with open(temp_path, mode) as f:
                    f.write(content)
            else:
                with open(temp_path, mode, encoding=encoding) as f:
                    f.write(content)
            
            # 成功したら本来のファイルに移動
            temp_path.replace(path)
            logger.debug(f"File safely written to {path}")
            
        except Exception as e:
            # エラー時は一時ファイルを削除
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to write file to {path}: {e}")
            raise