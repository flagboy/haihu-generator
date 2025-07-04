"""
ファイル入出力のための共通ユーティリティ
"""

import contextlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..core import (
    FileFormatError,
    FileReadError,
    FileWriteError,
    create_context,
)

logger = logging.getLogger(__name__)


class FileIOHelper:
    """ファイル入出力の共通処理を提供するヘルパークラス"""

    @staticmethod
    def save_json(
        data: dict[str, Any], path: str | Path, pretty: bool = True, ensure_ascii: bool = False
    ) -> None:
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
            "ensure_ascii": ensure_ascii,
            "separators": (",", ":") if not pretty else (",", ": "),
        }
        if pretty:
            json_kwargs["indent"] = 2

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, **json_kwargs)  # type: ignore[arg-type]
            logger.debug(f"JSON saved to {path}")
        except OSError as e:
            error_context = create_context(
                path=path, data_size=len(str(data)), pretty=pretty, ensure_ascii=ensure_ascii
            )
            raise FileWriteError(
                f"JSONファイルの保存に失敗しました: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e
        except (TypeError, ValueError) as e:
            error_context = create_context(
                path=path, data_type=type(data).__name__, pretty=pretty, ensure_ascii=ensure_ascii
            )
            raise FileFormatError(
                f"データをJSON形式に変換できません: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e

    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        """
        JSONファイルを読み込み

        Args:
            path: 読み込むファイルのパス

        Returns:
            読み込んだデータ
        """
        path = Path(path)

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"JSON loaded from {path}")
            return data  # type: ignore[no-any-return]
        except FileNotFoundError as e:
            raise FileReadError(
                f"JSONファイルが見つかりません: {path}",
                details={"path": str(path), "exists": path.exists()},
            ) from e
        except OSError as e:
            error_context = create_context(path=path)
            raise FileReadError(
                f"JSONファイルの読み込みに失敗しました: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e
        except json.JSONDecodeError as e:
            error_context = create_context(path=path, line=e.lineno, column=e.colno, position=e.pos)
            raise FileFormatError(
                f"無効なJSON形式です: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e

    @staticmethod
    def save_yaml(data: dict[str, Any], path: str | Path, default_flow_style: bool = False) -> None:
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
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=default_flow_style,
                    allow_unicode=True,
                    sort_keys=False,
                )
            logger.debug(f"YAML saved to {path}")
        except OSError as e:
            error_context = create_context(
                path=path, data_size=len(str(data)), default_flow_style=default_flow_style
            )
            raise FileWriteError(
                f"YAMLファイルの保存に失敗しました: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e
        except yaml.YAMLError as e:
            error_context = create_context(
                path=path, data_type=type(data).__name__, default_flow_style=default_flow_style
            )
            raise FileFormatError(
                f"データをYAML形式に変換できません: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e

    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        """
        YAMLファイルを読み込み

        Args:
            path: 読み込むファイルのパス

        Returns:
            読み込んだデータ
        """
        path = Path(path)

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            logger.debug(f"YAML loaded from {path}")
            return data
        except FileNotFoundError as e:
            raise FileReadError(
                f"YAMLファイルが見つかりません: {path}",
                details={"path": str(path), "exists": path.exists()},
            ) from e
        except OSError as e:
            error_context = create_context(path=path)
            raise FileReadError(
                f"YAMLファイルの読み込みに失敗しました: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e
        except yaml.YAMLError as e:
            error_context = create_context(path=path)
            raise FileFormatError(
                f"無効なYAML形式です: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e

    @staticmethod
    def save_pickle(data: Any, path: str | Path) -> None:
        """
        Pickleファイルを保存

        Args:
            data: 保存するデータ
            path: 保存先のパス
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Pickle saved to {path}")
        except OSError as e:
            error_context = create_context(path=path, data_type=type(data).__name__)
            raise FileWriteError(
                f"Pickleファイルの保存に失敗しました: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e
        except (pickle.PicklingError, TypeError) as e:
            error_context = create_context(path=path, data_type=type(data).__name__)
            raise FileFormatError(
                f"データをPickle形式に変換できません: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e

    @staticmethod
    def load_pickle(path: str | Path) -> Any:
        """
        Pickleファイルを読み込み

        Args:
            path: 読み込むファイルのパス

        Returns:
            読み込んだデータ
        """
        path = Path(path)

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"Pickle loaded from {path}")
            return data
        except FileNotFoundError as e:
            raise FileReadError(
                f"Pickleファイルが見つかりません: {path}",
                details={"path": str(path), "exists": path.exists()},
            ) from e
        except OSError as e:
            error_context = create_context(path=path)
            raise FileReadError(
                f"Pickleファイルの読み込みに失敗しました: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e
        except (pickle.UnpicklingError, ValueError, EOFError) as e:
            error_context = create_context(path=path)
            raise FileFormatError(
                f"無効なPickle形式または破損したファイルです: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e

    @staticmethod
    def ensure_directory(path: str | Path) -> Path:
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
    def safe_write(
        path: str | Path, content: str | bytes, mode: str = "w", encoding: str | None = "utf-8"
    ) -> None:
        """
        安全なファイル書き込み（一時ファイル経由）

        Args:
            path: 書き込み先パス
            content: 書き込む内容
            mode: ファイルモード
            encoding: エンコーディング（テキストモードの場合）
        """
        path = Path(path)
        temp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            # 一時ファイルに書き込み
            if "b" in mode:
                with open(temp_path, mode) as f:
                    f.write(content)
            else:
                with open(temp_path, mode, encoding=encoding) as f:
                    f.write(content)

            # 成功したら本来のファイルに移動
            temp_path.replace(path)
            logger.debug(f"File safely written to {path}")

        except OSError as e:
            # エラー時は一時ファイルを削除
            if temp_path.exists():
                with contextlib.suppress(Exception):
                    temp_path.unlink()  # 一時ファイルの削除に失敗しても無視

            error_context = create_context(
                path=path,
                mode=mode,
                encoding=encoding,
                content_size=len(content) if isinstance(content, str | bytes) else "unknown",
            )
            raise FileWriteError(
                f"ファイルの安全な書き込みに失敗しました: {path}",
                details={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "temp_path": str(temp_path),
                    **error_context,
                },
            ) from e
        except Exception as e:
            # エラー時は一時ファイルを削除
            if temp_path.exists():
                with contextlib.suppress(Exception):
                    temp_path.unlink()

            error_context = create_context(path=path, mode=mode, encoding=encoding)
            raise FileWriteError(
                f"予期しないエラーが発生しました: {path}",
                details={"original_error": str(e), "error_type": type(e).__name__, **error_context},
            ) from e
