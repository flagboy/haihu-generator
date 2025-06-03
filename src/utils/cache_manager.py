"""
キャッシュ管理ユーティリティ
システム全体で使用されるキャッシュ機能を統一管理
"""

import json
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logger import LoggerMixin


@dataclass
class CacheEntry:
    """キャッシュエントリ"""

    key: str
    value: Any
    timestamp: float
    ttl: float | None = None  # Time To Live (秒)

    def is_expired(self) -> bool:
        """有効期限切れかチェック"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class CacheBackend(ABC):
    """キャッシュバックエンドの抽象基底クラス"""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """キャッシュから値を取得"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None = None):
        """キャッシュに値を設定"""
        pass

    @abstractmethod
    def delete(self, key: str):
        """キャッシュから値を削除"""
        pass

    @abstractmethod
    def clear(self):
        """キャッシュをクリア"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """キーが存在するかチェック"""
        pass


class MemoryCacheBackend(CacheBackend):
    """メモリベースのキャッシュバックエンド"""

    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, CacheEntry] = {}
        self.max_size = max_size

    def get(self, key: str) -> Any | None:
        """キャッシュから値を取得"""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if entry.is_expired():
            self.delete(key)
            return None

        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None):
        """キャッシュに値を設定"""
        # サイズ制限チェック
        if len(self._cache) >= self.max_size and key not in self._cache:
            # 最も古いエントリを削除（LRU風）
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            self.delete(oldest_key)

        self._cache[key] = CacheEntry(key=key, value=value, timestamp=time.time(), ttl=ttl)

    def delete(self, key: str):
        """キャッシュから値を削除"""
        self._cache.pop(key, None)

    def clear(self):
        """キャッシュをクリア"""
        self._cache.clear()

    def exists(self, key: str) -> bool:
        """キーが存在するかチェック"""
        if key not in self._cache:
            return False

        entry = self._cache[key]
        if entry.is_expired():
            self.delete(key)
            return False

        return True

    def size(self) -> int:
        """キャッシュサイズを取得"""
        return len(self._cache)


class FileCacheBackend(CacheBackend, LoggerMixin):
    """ファイルベースのキャッシュバックエンド"""

    def __init__(self, cache_dir: str | Path):
        """
        初期化

        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.cache_dir / "_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        """メタデータを読み込み"""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load cache metadata: {e}")
        return {}

    def _save_metadata(self):
        """メタデータを保存"""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")

    def _get_cache_file(self, key: str) -> Path:
        """キャッシュファイルパスを取得"""
        # キーをファイル名として使用（特殊文字を置換）
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.pkl"

    def get(self, key: str) -> Any | None:
        """キャッシュから値を取得"""
        if key not in self._metadata:
            return None

        metadata = self._metadata[key]
        ttl = metadata.get("ttl")
        timestamp = metadata.get("timestamp", 0)

        if ttl is not None and time.time() - timestamp > ttl:
            self.delete(key)
            return None

        cache_file = self._get_cache_file(key)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load cache for key '{key}': {e}")
            return None

    def set(self, key: str, value: Any, ttl: float | None = None):
        """キャッシュに値を設定"""
        cache_file = self._get_cache_file(key)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)

            self._metadata[key] = {"timestamp": time.time(), "ttl": ttl, "file": cache_file.name}
            self._save_metadata()
        except Exception as e:
            self.logger.error(f"Failed to save cache for key '{key}': {e}")

    def delete(self, key: str):
        """キャッシュから値を削除"""
        if key in self._metadata:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.error(f"Failed to delete cache file: {e}")

            del self._metadata[key]
            self._save_metadata()

    def clear(self):
        """キャッシュをクリア"""
        for key in list(self._metadata.keys()):
            self.delete(key)

        self._metadata.clear()
        self._save_metadata()

    def exists(self, key: str) -> bool:
        """キーが存在するかチェック"""
        if key not in self._metadata:
            return False

        metadata = self._metadata[key]
        ttl = metadata.get("ttl")
        timestamp = metadata.get("timestamp", 0)

        if ttl is not None and time.time() - timestamp > ttl:
            self.delete(key)
            return False

        return self._get_cache_file(key).exists()


class CacheManager(LoggerMixin):
    """キャッシュマネージャー"""

    def __init__(self, backend: CacheBackend | None = None, default_ttl: float | None = None):
        """
        初期化

        Args:
            backend: キャッシュバックエンド
            default_ttl: デフォルトTTL（秒）
        """
        self.backend = backend or MemoryCacheBackend()
        self.default_ttl = default_ttl
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
        self.logger.info(f"CacheManager initialized with {type(self.backend).__name__}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        キャッシュから値を取得

        Args:
            key: キー
            default: デフォルト値

        Returns:
            キャッシュされた値またはデフォルト値
        """
        value = self.backend.get(key)
        if value is not None:
            self._stats["hits"] += 1
            return value
        else:
            self._stats["misses"] += 1
            return default

    def set(self, key: str, value: Any, ttl: float | None = None):
        """
        キャッシュに値を設定

        Args:
            key: キー
            value: 値
            ttl: TTL（秒）
        """
        if ttl is None:
            ttl = self.default_ttl

        self.backend.set(key, value, ttl)
        self._stats["sets"] += 1

    def delete(self, key: str):
        """キャッシュから値を削除"""
        self.backend.delete(key)
        self._stats["deletes"] += 1

    def clear(self):
        """キャッシュをクリア"""
        self.backend.clear()
        self.logger.info("Cache cleared")

    def exists(self, key: str) -> bool:
        """キーが存在するかチェック"""
        return self.backend.exists(key)

    def cache_decorator(self, key_func: Callable | None = None, ttl: float | None = None):
        """
        キャッシュデコレータ

        Args:
            key_func: キー生成関数
            ttl: TTL（秒）

        Returns:
            デコレータ関数
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # キー生成
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # デフォルトのキー生成
                    cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

                # キャッシュチェック
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # 関数実行
                result = func(*args, **kwargs)

                # キャッシュ保存
                self.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def get_stats(self) -> dict[str, int]:
        """統計情報を取得"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {**self._stats, "total_requests": total, "hit_rate": hit_rate}

    def reset_stats(self):
        """統計情報をリセット"""
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}


# グローバルキャッシュインスタンス
_global_cache: CacheManager | None = None


def get_global_cache() -> CacheManager:
    """グローバルキャッシュインスタンスを取得"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def set_global_cache(cache: CacheManager):
    """グローバルキャッシュインスタンスを設定"""
    global _global_cache
    _global_cache = cache
