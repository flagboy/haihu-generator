"""
処理結果の保存と変換を担当するクラス
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.file_io import FileIOHelper
from ..output.tenhou_json_formatter import TenhouJsonFormatter
from ..models.tenhou_game_data import TenhouGameData


class ResultProcessor:
    """処理結果の保存と変換を行うクラス"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        初期化
        
        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.logger = get_logger(self.__class__.__name__)
        
        # 天鳳JSON設定
        self.tenhou_config = self.config.get('tenhou_json', {})
        self.formatter = TenhouJsonFormatter(config_manager)
        
    def save_results(self,
                    game_data: Any,
                    output_path: str,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        処理結果を保存
        
        Args:
            game_data: ゲームデータ
            output_path: 出力パス
            metadata: メタデータ（オプション）
        """
        try:
            # 天鳳形式に変換
            if hasattr(game_data, 'to_tenhou_format'):
                tenhou_data = game_data.to_tenhou_format()
            else:
                # TenhouGameDataオブジェクトに変換
                tenhou_game = TenhouGameData()
                self._populate_tenhou_data(tenhou_game, game_data)
                tenhou_data = self.formatter.format_game_data(tenhou_game)
            
            # メタデータを追加
            if metadata:
                tenhou_data['metadata'] = metadata
            
            # 最適化処理
            tenhou_data = self._optimize_tenhou_data(tenhou_data)
            
            # ファイル保存
            pretty_print = self.tenhou_config.get('pretty_print', True)
            FileIOHelper.save_json(tenhou_data, output_path, pretty=pretty_print)
            
            self.logger.info(f"結果を保存しました: {output_path}")
            
        except Exception as e:
            self.logger.error(f"結果の保存に失敗しました: {e}", exc_info=True)
            raise
    
    def _populate_tenhou_data(self,
                             tenhou_game: TenhouGameData,
                             game_data: Any) -> None:
        """
        ゲームデータをTenhouGameDataに変換
        
        Args:
            tenhou_game: 天鳳ゲームデータオブジェクト
            game_data: 元のゲームデータ
        """
        # ゲーム情報の設定
        if hasattr(game_data, 'game_info'):
            info = game_data.game_info
            tenhou_game.set_game_info(
                room_name=info.get('room_name', 'Unknown'),
                game_type=info.get('game_type', '四人打ち'),
                rules=info.get('rules', {})
            )
        
        # プレイヤー情報の設定
        if hasattr(game_data, 'players'):
            for i, player in enumerate(game_data.players):
                tenhou_game.set_player_info(
                    seat=i,
                    name=player.get('name', f'Player{i+1}'),
                    initial_score=player.get('initial_score', 25000),
                    rank=player.get('rank', '初段')
                )
        
        # ラウンド情報の追加
        if hasattr(game_data, 'rounds'):
            for round_data in game_data.rounds:
                self._add_round_data(tenhou_game, round_data)
    
    def _add_round_data(self,
                       tenhou_game: TenhouGameData,
                       round_data: Dict[str, Any]) -> None:
        """
        ラウンドデータを追加
        
        Args:
            tenhou_game: 天鳳ゲームデータオブジェクト
            round_data: ラウンドデータ
        """
        round_num = round_data.get('round_number', 0)
        honba = round_data.get('honba', 0)
        
        tenhou_game.start_new_round(
            round_number=round_num,
            honba=honba,
            riichi_sticks=round_data.get('riichi_sticks', 0),
            dora_indicators=round_data.get('dora_indicators', [])
        )
        
        # アクションの追加
        for action in round_data.get('actions', []):
            self._add_action(tenhou_game, action)
    
    def _add_action(self,
                   tenhou_game: TenhouGameData,
                   action: Dict[str, Any]) -> None:
        """
        アクションを追加
        
        Args:
            tenhou_game: 天鳳ゲームデータオブジェクト
            action: アクションデータ
        """
        action_type = action.get('type')
        player = action.get('player', 0)
        
        if action_type == 'draw':
            tenhou_game.add_draw_action(
                player=player,
                tile=action.get('tile')
            )
        elif action_type == 'discard':
            tenhou_game.add_discard_action(
                player=player,
                tile=action.get('tile'),
                is_riichi=action.get('is_riichi', False),
                is_tsumogiri=action.get('is_tsumogiri', False)
            )
        elif action_type in ['chi', 'pon', 'kan']:
            tenhou_game.add_call_action(
                player=player,
                action_type=action_type,
                tiles=action.get('tiles', []),
                from_player=action.get('from_player', (player + 3) % 4),
                called_tile=action.get('called_tile')
            )
    
    def _optimize_tenhou_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        天鳳データを最適化
        
        Args:
            data: 天鳳形式のデータ
            
        Returns:
            最適化されたデータ
        """
        optimization = self.tenhou_config.get('optimization', {})
        
        if optimization.get('remove_empty_fields', True):
            data = self._remove_empty_fields(data)
        
        if optimization.get('compress_redundant_data', True):
            data = self._compress_redundant_data(data)
        
        return data
    
    def _remove_empty_fields(self, data: Any) -> Any:
        """
        空のフィールドを削除
        
        Args:
            data: 処理対象のデータ
            
        Returns:
            空フィールドを削除したデータ
        """
        if isinstance(data, dict):
            return {k: self._remove_empty_fields(v) 
                   for k, v in data.items() 
                   if v is not None and v != "" and v != [] and v != {}}
        elif isinstance(data, list):
            return [self._remove_empty_fields(item) for item in data]
        else:
            return data
    
    def _compress_redundant_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        冗長なデータを圧縮
        
        Args:
            data: 処理対象のデータ
            
        Returns:
            圧縮されたデータ
        """
        # 連続する同じアクションをまとめる
        if 'rounds' in data:
            for round_data in data['rounds']:
                if 'actions' in round_data:
                    round_data['actions'] = self._compress_actions(round_data['actions'])
        
        return data
    
    def _compress_actions(self, actions: List[Any]) -> List[Any]:
        """
        アクションリストを圧縮
        
        Args:
            actions: アクションリスト
            
        Returns:
            圧縮されたアクションリスト
        """
        if not actions:
            return actions
        
        compressed = []
        current_group = None
        
        for action in actions:
            # 同じ種類の連続したアクションをグループ化
            if (current_group and 
                action[0] == current_group['type'] and 
                len(action) == 2):  # シンプルなアクションのみ
                current_group['actions'].append(action)
            else:
                if current_group and len(current_group['actions']) > 1:
                    # グループ化されたアクションを追加
                    compressed.append([current_group['type'], current_group['actions']])
                elif current_group:
                    # 単一のアクションはそのまま追加
                    compressed.extend(current_group['actions'])
                
                # 新しいグループを開始
                if len(action) == 2 and isinstance(action[0], str):
                    current_group = {
                        'type': action[0],
                        'actions': [action]
                    }
                else:
                    compressed.append(action)
                    current_group = None
        
        # 最後のグループを処理
        if current_group:
            if len(current_group['actions']) > 1:
                compressed.append([current_group['type'], current_group['actions']])
            else:
                compressed.extend(current_group['actions'])
        
        return compressed
    
    def export_statistics(self,
                         statistics: Dict[str, Any],
                         output_path: str) -> None:
        """
        統計情報をエクスポート
        
        Args:
            statistics: 統計情報
            output_path: 出力パス
        """
        stats_path = Path(output_path).with_suffix('.stats.json')
        FileIOHelper.save_json(statistics, stats_path, pretty=True)
        self.logger.info(f"統計情報を保存しました: {stats_path}")