"""
天鳳形式専用バリデーター
天鳳JSON仕様準拠チェックとデータ整合性検証機能を提供
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from ..models.tenhou_game_data import TenhouAction, TenhouActionType, TenhouGameData
from ..utils.logger import get_logger
from ..utils.tile_definitions import TileDefinitions


@dataclass
class ValidationResult:
    """バリデーション結果"""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    score: float  # 0.0-1.0の品質スコア

    def add_error(self, message: str) -> None:
        """エラーを追加"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """警告を追加"""
        self.warnings.append(message)

    def calculate_score(self) -> None:
        """品質スコアを計算"""
        if not self.is_valid:
            self.score = 0.0
        else:
            # 警告数に基づいてスコアを減点
            penalty = min(len(self.warnings) * 0.05, 0.5)
            self.score = max(0.5, 1.0 - penalty)


class TenhouValidator:
    """天鳳形式専用バリデーター"""

    # 天鳳記法の正規表現パターン
    TENHOU_TILE_PATTERN = re.compile(r"^[0-9][mps]|[1-7]z$")
    TENHOU_ACTION_PATTERN = re.compile(r"^([TD][0-3]|N[0-3]|REACH[0-3]|AGARI[0-3]|RYUU)$")

    def __init__(self):
        """バリデーターの初期化"""
        self.tile_definitions = TileDefinitions()
        self.logger = get_logger(__name__)
        self._validation_cache: dict[str, ValidationResult] = {}

    def validate_tenhou_json(self, json_data: str) -> ValidationResult:
        """天鳳JSON形式の妥当性を検証

        Args:
            json_data: 検証する天鳳JSON文字列

        Returns:
            ValidationResult: 検証結果
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], score=1.0)

        try:
            # JSON形式の検証
            data = json.loads(json_data)

            # 基本構造の検証
            self._validate_basic_structure(data, result)

            # 各フィールドの詳細検証
            if result.is_valid:
                self._validate_title(data.get("title", ""), result)
                self._validate_player_names(data.get("name", []), result)
                self._validate_game_rules(data.get("rule", {}), result)
                self._validate_game_log(data.get("log", []), result)
                self._validate_scores(data.get("sc", []), result)
                self._validate_result(data.get("owari", {}), result)

            # 整合性チェック
            if result.is_valid:
                self._validate_consistency(data, result)

        except json.JSONDecodeError as e:
            result.add_error(f"JSON形式エラー: {e}")
        except Exception as e:
            result.add_error(f"予期しないエラー: {e}")

        result.calculate_score()
        return result

    def validate_tenhou_game_data(self, game_data: TenhouGameData) -> ValidationResult:
        """天鳳ゲームデータオブジェクトの妥当性を検証

        Args:
            game_data: 検証するゲームデータ

        Returns:
            ValidationResult: 検証結果
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], score=1.0)

        # データ構造の基本検証
        structure_valid, structure_errors = game_data.validate_structure()
        if not structure_valid:
            for error in structure_errors:
                result.add_error(error)

        if result.is_valid:
            # アクションシーケンスの検証
            self._validate_action_sequence(game_data.actions, result)

            # プレイヤー状態の検証
            self._validate_player_states(game_data.players, result)

            # ゲームルールの検証
            self._validate_game_rule_consistency(game_data, result)

        result.calculate_score()
        return result

    def _validate_basic_structure(self, data: dict[str, Any], result: ValidationResult) -> None:
        """基本構造の検証"""
        required_fields = ["title", "name", "rule", "log", "sc", "owari"]

        for field in required_fields:
            if field not in data:
                result.add_error(f"必須フィールド '{field}' が見つかりません")

        # データ型の検証
        if "name" in data and not isinstance(data["name"], list):
            result.add_error("'name'フィールドはリスト形式である必要があります")

        if "rule" in data and not isinstance(data["rule"], dict):
            result.add_error("'rule'フィールドは辞書形式である必要があります")

        if "log" in data and not isinstance(data["log"], list):
            result.add_error("'log'フィールドはリスト形式である必要があります")

        if "sc" in data and not isinstance(data["sc"], list):
            result.add_error("'sc'フィールドはリスト形式である必要があります")

    def _validate_title(self, title: str, result: ValidationResult) -> None:
        """タイトルの検証"""
        if not title or not isinstance(title, str):
            result.add_error("タイトルが設定されていないか、文字列ではありません")
        elif len(title.strip()) == 0:
            result.add_warning("タイトルが空文字列です")

    def _validate_player_names(self, names: list[str], result: ValidationResult) -> None:
        """プレイヤー名の検証"""
        if len(names) != 4:
            result.add_error(f"プレイヤー名は4つである必要があります（現在: {len(names)}）")

        for i, name in enumerate(names):
            if not isinstance(name, str):
                result.add_error(f"プレイヤー{i + 1}の名前が文字列ではありません")
            elif len(name.strip()) == 0:
                result.add_warning(f"プレイヤー{i + 1}の名前が空です")

    def _validate_game_rules(self, rules: dict[str, Any], result: ValidationResult) -> None:
        """ゲームルールの検証"""
        expected_fields = ["disp", "aka", "kuitan", "tonnan"]

        for field in expected_fields:
            if field not in rules:
                result.add_warning(f"ルールフィールド '{field}' が見つかりません")

        # 各フィールドの値検証
        if "aka" in rules and rules["aka"] not in [0, 1]:
            result.add_error("'aka'フィールドは0または1である必要があります")

        if "kuitan" in rules and rules["kuitan"] not in [0, 1]:
            result.add_error("'kuitan'フィールドは0または1である必要があります")

        if "tonnan" in rules and rules["tonnan"] not in [0, 1]:
            result.add_error("'tonnan'フィールドは0または1である必要があります")

    def _validate_game_log(self, log: list[list[Any]], result: ValidationResult) -> None:
        """ゲームログの検証"""
        if not log:
            result.add_warning("ゲームログが空です")
            return

        for i, action in enumerate(log):
            if not isinstance(action, list):
                result.add_error(f"ログエントリ{i}がリスト形式ではありません")
                continue

            if len(action) == 0:
                result.add_error(f"ログエントリ{i}が空です")
                continue

            # アクション形式の検証
            self._validate_single_action(action, i, result)

    def _validate_single_action(
        self, action: list[Any], index: int, result: ValidationResult
    ) -> None:
        """単一アクションの検証"""
        action_type = str(action[0]) if action else ""

        if not self.TENHOU_ACTION_PATTERN.match(action_type):
            result.add_error(f"ログエントリ{index}: 不正なアクション形式 '{action_type}'")
            return

        # アクション種別に応じた詳細検証
        if action_type.startswith("T"):  # ツモ
            self._validate_draw_action(action, index, result)
        elif action_type.startswith("D"):  # 打牌
            self._validate_discard_action(action, index, result)
        elif action_type.startswith("N"):  # 鳴き
            self._validate_call_action(action, index, result)
        elif action_type.startswith("REACH"):  # リーチ
            self._validate_riichi_action(action, index, result)
        elif action_type.startswith("AGARI"):  # 和了
            self._validate_agari_action(action, index, result)

    def _validate_draw_action(
        self, action: list[Any], index: int, result: ValidationResult
    ) -> None:
        """ツモアクションの検証"""
        if len(action) < 2:
            result.add_error(f"ログエントリ{index}: ツモアクションの引数が不足しています")
            return

        tile = str(action[1])
        if not self._is_valid_tenhou_tile(tile):
            result.add_error(f"ログエントリ{index}: 不正な牌記法 '{tile}'")

    def _validate_discard_action(
        self, action: list[Any], index: int, result: ValidationResult
    ) -> None:
        """打牌アクションの検証"""
        if len(action) < 2:
            result.add_error(f"ログエントリ{index}: 打牌アクションの引数が不足しています")
            return

        tile = str(action[1])
        if not self._is_valid_tenhou_tile(tile):
            result.add_error(f"ログエントリ{index}: 不正な牌記法 '{tile}'")

        # リーチフラグの検証
        if len(action) > 2:
            for flag in action[2:]:
                if flag not in ["r", "t"]:  # r: リーチ, t: ツモ切り
                    result.add_warning(f"ログエントリ{index}: 不明なフラグ '{flag}'")

    def _validate_call_action(
        self, action: list[Any], index: int, result: ValidationResult
    ) -> None:
        """鳴きアクションの検証"""
        if len(action) < 3:
            result.add_error(f"ログエントリ{index}: 鳴きアクションの引数が不足しています")
            return

        call_type = str(action[1])
        if call_type not in ["chi", "pon", "kan", "ankan"]:
            result.add_error(f"ログエントリ{index}: 不正な鳴き種別 '{call_type}'")

        if len(action) > 2 and isinstance(action[2], list):
            for tile in action[2]:
                if not self._is_valid_tenhou_tile(str(tile)):
                    result.add_error(f"ログエントリ{index}: 不正な牌記法 '{tile}'")

    def _validate_riichi_action(
        self, action: list[Any], index: int, result: ValidationResult
    ) -> None:
        """リーチアクションの検証"""
        if len(action) < 2:
            result.add_warning(f"ログエントリ{index}: リーチアクションにステップ情報がありません")
        elif not isinstance(action[1], int) or action[1] not in [1, 2]:
            result.add_warning(f"ログエントリ{index}: リーチステップが不正です")

    def _validate_agari_action(
        self, action: list[Any], index: int, result: ValidationResult
    ) -> None:
        """和了アクションの検証"""
        if len(action) < 5:
            result.add_error(f"ログエントリ{index}: 和了アクションの引数が不足しています")
            return

        agari_type = str(action[1])
        if not (agari_type == "tsumo" or agari_type.startswith("ron")):
            result.add_error(f"ログエントリ{index}: 不正な和了種別 '{agari_type}'")

        # 翻数、符数、得点の検証
        try:
            han = int(action[2])
            fu = int(action[3])
            score = int(action[4])

            if han < 0 or han > 13:
                result.add_warning(f"ログエントリ{index}: 翻数が範囲外です ({han})")
            if fu < 0 or fu > 110:
                result.add_warning(f"ログエントリ{index}: 符数が範囲外です ({fu})")
            if score < 0:
                result.add_warning(f"ログエントリ{index}: 得点が負の値です ({score})")
        except (ValueError, IndexError):
            result.add_error(f"ログエントリ{index}: 翻数、符数、得点が数値ではありません")

    def _validate_scores(self, scores: list[int], result: ValidationResult) -> None:
        """スコアの検証"""
        if len(scores) != 4:
            result.add_error(f"スコアは4つである必要があります（現在: {len(scores)}）")

        for i, score in enumerate(scores):
            if not isinstance(score, int):
                result.add_error(f"プレイヤー{i + 1}のスコアが整数ではありません")
            elif score < 0:
                result.add_warning(f"プレイヤー{i + 1}のスコアが負の値です ({score})")

    def _validate_result(self, result_data: dict[str, Any], result: ValidationResult) -> None:
        """結果データの検証"""
        if not result_data:
            result.add_warning("結果データが空です")
            return

        expected_fields = ["順位", "得点", "ウマ"]
        for field in expected_fields:
            if field in result_data:
                if not isinstance(result_data[field], list):
                    result.add_error(f"結果フィールド '{field}' がリスト形式ではありません")
                elif len(result_data[field]) != 4:
                    result.add_error(f"結果フィールド '{field}' の要素数が4ではありません")

    def _validate_consistency(self, data: dict[str, Any], result: ValidationResult) -> None:
        """データ整合性の検証"""
        # プレイヤー数とスコア数の整合性
        player_count = len(data.get("name", []))
        score_count = len(data.get("sc", []))

        if player_count != score_count:
            result.add_error(f"プレイヤー数({player_count})とスコア数({score_count})が一致しません")

        # ログとスコアの整合性チェック
        self._validate_log_score_consistency(data.get("log", []), data.get("sc", []), result)

    def _validate_log_score_consistency(
        self, log: list[list[Any]], scores: list[int], result: ValidationResult
    ) -> None:
        """ログとスコアの整合性検証"""
        # 簡易的な整合性チェック
        if not log or not scores:
            return

        # 和了アクションの得点とスコアの関係をチェック
        total_score_changes = 0
        for action in log:
            if len(action) > 0 and str(action[0]).startswith("AGARI") and len(action) >= 5:
                try:
                    score_change = int(action[4])
                    total_score_changes += score_change
                except (ValueError, IndexError):
                    pass

        # 基本的な得点範囲チェック
        total_score = sum(scores)
        expected_total = 100000  # 4人 × 25000点

        if abs(total_score - expected_total) > 1000:  # 1000点の誤差を許容
            result.add_warning(
                f"総得点が期待値と大きく異なります (実際: {total_score}, 期待: {expected_total})"
            )

    def _validate_action_sequence(
        self, actions: list[TenhouAction], result: ValidationResult
    ) -> None:
        """アクションシーケンスの検証"""
        if not actions:
            result.add_warning("アクションが空です")
            return

        # アクションの順序性チェック
        last_player = -1
        for i, action in enumerate(actions):
            # プレイヤーIDの妥当性
            if action.player < 0 or action.player >= 4:
                result.add_error(f"アクション{i}: 不正なプレイヤーID {action.player}")

            # 基本的な順序チェック（簡易版）
            if (
                action.action_type == TenhouActionType.DRAW
                and last_player != -1
                and action.player != (last_player + 1) % 4
            ):
                result.add_warning(f"アクション{i}: ツモ順序が不自然です")

            last_player = action.player

    def _validate_player_states(self, players: list[Any], result: ValidationResult) -> None:
        """プレイヤー状態の検証"""
        if len(players) != 4:
            result.add_error(f"プレイヤー数が4人ではありません: {len(players)}")

        for i, player in enumerate(players):
            if hasattr(player, "score") and player.score < 0:
                result.add_warning(f"プレイヤー{i + 1}のスコアが負の値です: {player.score}")

            if hasattr(player, "hand") and len(player.hand) > 14:
                result.add_error(f"プレイヤー{i + 1}の手牌が14枚を超えています: {len(player.hand)}")

    def _validate_game_rule_consistency(
        self, game_data: TenhouGameData, result: ValidationResult
    ) -> None:
        """ゲームルール整合性の検証"""
        # 赤ドラ設定と実際の牌使用の整合性
        if not game_data.rule.red_dora:
            for action in game_data.actions:
                if (
                    hasattr(action, "tile")
                    and hasattr(action.tile, "is_red_dora")
                    and action.tile.is_red_dora
                ):
                    result.add_warning("赤ドラ無しルールなのに赤ドラが使用されています")
                    break

    def _is_valid_tenhou_tile(self, tile: str) -> bool:
        """天鳳記法の牌が有効かどうか判定"""
        if not tile or len(tile) != 2:
            return False
        return bool(self.TENHOU_TILE_PATTERN.match(tile))

    def validate_tile_notation(self, tile: str) -> bool:
        """牌記法の妥当性を検証"""
        return self.tile_definitions.is_tenhou_notation(tile)

    def get_validation_summary(self, result: ValidationResult) -> str:
        """検証結果のサマリーを取得"""
        summary = f"検証結果: {'合格' if result.is_valid else '不合格'}\n"
        summary += f"品質スコア: {result.score:.2f}\n"

        if result.errors:
            summary += f"エラー数: {len(result.errors)}\n"
            for error in result.errors[:5]:  # 最初の5件のみ表示
                summary += f"  - {error}\n"
            if len(result.errors) > 5:
                summary += f"  ... 他{len(result.errors) - 5}件\n"

        if result.warnings:
            summary += f"警告数: {len(result.warnings)}\n"
            for warning in result.warnings[:3]:  # 最初の3件のみ表示
                summary += f"  - {warning}\n"
            if len(result.warnings) > 3:
                summary += f"  ... 他{len(result.warnings) - 3}件\n"

        return summary

    def clear_cache(self) -> None:
        """バリデーションキャッシュをクリア"""
        self._validation_cache.clear()
        self.logger.debug("バリデーションキャッシュをクリアしました")
