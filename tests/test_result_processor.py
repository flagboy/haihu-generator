"""
ResultProcessorのテスト
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.integration.result_processor import ResultProcessor
from src.utils.config import ConfigManager
from src.models.tenhou_game_data import TenhouGameData


class TestResultProcessor:
    """ResultProcessorのテストクラス"""
    
    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのモック"""
        config_manager = Mock(spec=ConfigManager)
        config_manager.config = {
            'tenhou_json': {
                'pretty_print': True,
                'optimization': {
                    'remove_empty_fields': True,
                    'compress_redundant_data': True
                }
            }
        }
        return config_manager
    
    @pytest.fixture
    def result_processor(self, config_manager):
        """ResultProcessorのフィクスチャ"""
        return ResultProcessor(config_manager)
    
    @pytest.fixture
    def sample_game_data(self):
        """サンプルゲームデータ"""
        game_data = Mock()
        game_data.to_tenhou_format.return_value = {
            'title': '天鳳牌譜テスト',
            'players': [
                {'name': 'プレイヤー1', 'score': 25000},
                {'name': 'プレイヤー2', 'score': 25000},
                {'name': 'プレイヤー3', 'score': 25000},
                {'name': 'プレイヤー4', 'score': 25000}
            ],
            'rounds': [
                {
                    'round_number': 0,
                    'actions': [
                        ['T0', '1m'],
                        ['D0', '9m'],
                        ['T1', '5p']
                    ]
                }
            ]
        }
        return game_data
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリのフィクスチャ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_initialization(self, result_processor):
        """初期化テスト"""
        assert result_processor is not None
        assert result_processor.tenhou_config is not None
        assert result_processor.formatter is not None
    
    def test_save_results_with_tenhou_format(self, result_processor, sample_game_data, temp_dir):
        """天鳳形式での結果保存テスト"""
        output_path = os.path.join(temp_dir, 'result.json')
        metadata = {
            'video_path': '/path/to/video.mp4',
            'processing_time': 123.45,
            'frame_count': 1000
        }
        
        # 保存実行
        result_processor.save_results(sample_game_data, output_path, metadata)
        
        # ファイルが作成されたことを確認
        assert os.path.exists(output_path)
        
        # 内容を確認
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data['title'] == '天鳳牌譜テスト'
        assert len(saved_data['players']) == 4
        assert saved_data['metadata'] == metadata
    
    def test_save_results_without_metadata(self, result_processor, sample_game_data, temp_dir):
        """メタデータなしでの保存テスト"""
        output_path = os.path.join(temp_dir, 'result_no_meta.json')
        
        # 保存実行
        result_processor.save_results(sample_game_data, output_path)
        
        # ファイルが作成されたことを確認
        assert os.path.exists(output_path)
        
        # メタデータが含まれていないことを確認
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert 'metadata' not in saved_data
    
    def test_populate_tenhou_data(self, result_processor):
        """TenhouGameDataへの変換テスト"""
        # ゲームデータのモック
        game_data = Mock()
        game_data.game_info = {
            'room_name': 'テストルーム',
            'game_type': '四人打ち',
            'rules': {'aka': True}
        }
        game_data.players = [
            {'name': 'Player1', 'initial_score': 25000, 'rank': '三段'},
            {'name': 'Player2', 'initial_score': 25000, 'rank': '初段'}
        ]
        game_data.rounds = [
            {
                'round_number': 0,
                'honba': 0,
                'riichi_sticks': 0,
                'dora_indicators': ['1m'],
                'actions': [
                    {'type': 'draw', 'player': 0, 'tile': '5m'},
                    {'type': 'discard', 'player': 0, 'tile': '9s', 'is_riichi': False}
                ]
            }
        ]
        
        # TenhouGameDataのモック
        with patch('src.integration.result_processor.TenhouGameData') as mock_tenhou_class:
            mock_tenhou = Mock()
            mock_tenhou_class.return_value = mock_tenhou
            
            # 変換実行
            result_processor._populate_tenhou_data(mock_tenhou, game_data)
            
            # メソッドが適切に呼び出されたか確認
            mock_tenhou.set_game_info.assert_called_once_with(
                room_name='テストルーム',
                game_type='四人打ち',
                rules={'aka': True}
            )
            
            assert mock_tenhou.set_player_info.call_count == 2
            mock_tenhou.start_new_round.assert_called_once()
            mock_tenhou.add_draw_action.assert_called_once()
            mock_tenhou.add_discard_action.assert_called_once()
    
    def test_optimize_tenhou_data(self, result_processor):
        """天鳳データ最適化のテスト"""
        # 最適化前のデータ
        data = {
            'title': 'テスト',
            'empty_field': '',
            'null_field': None,
            'empty_list': [],
            'empty_dict': {},
            'valid_field': 'value',
            'nested': {
                'empty': None,
                'valid': 'nested_value'
            },
            'rounds': [
                {
                    'actions': [
                        ['D0', '1m'],
                        ['D0', '2m'],
                        ['D0', '3m'],
                        ['T1', '5p']
                    ]
                }
            ]
        }
        
        # 最適化実行
        optimized = result_processor._optimize_tenhou_data(data)
        
        # 空フィールドが削除されていることを確認
        assert 'empty_field' not in optimized
        assert 'null_field' not in optimized
        assert 'empty_list' not in optimized
        assert 'empty_dict' not in optimized
        assert 'valid_field' in optimized
        assert 'empty' not in optimized['nested']
        assert optimized['nested']['valid'] == 'nested_value'
    
    def test_remove_empty_fields(self, result_processor):
        """空フィールド削除のテスト"""
        data = {
            'a': 'value',
            'b': '',
            'c': None,
            'd': [],
            'e': {},
            'f': {
                'g': 'nested',
                'h': None,
                'i': []
            },
            'j': [1, 2, 3],
            'k': {'l': 'value'}
        }
        
        result = result_processor._remove_empty_fields(data)
        
        # 期待される結果
        assert result == {
            'a': 'value',
            'f': {
                'g': 'nested'
            },
            'j': [1, 2, 3],
            'k': {'l': 'value'}
        }
    
    def test_compress_redundant_data(self, result_processor):
        """冗長データ圧縮のテスト"""
        data = {
            'rounds': [
                {
                    'actions': [
                        ['D0', '1m'],
                        ['D0', '2m'],
                        ['D0', '3m'],
                        ['T1', '5p'],
                        ['T1', '6p'],
                        ['D1', '9s']
                    ]
                }
            ]
        }
        
        compressed = result_processor._compress_redundant_data(data)
        
        # アクションが圧縮されていることを確認
        actions = compressed['rounds'][0]['actions']
        
        # D0の連続したアクションがグループ化されているか確認
        # （実装に応じて詳細なアサーションを追加）
        assert len(actions) < 6  # 元の6アクションより少ない
    
    def test_compress_actions(self, result_processor):
        """アクション圧縮の詳細テスト"""
        actions = [
            ['D0', '1m'],
            ['D0', '2m'],
            ['D0', '3m'],
            ['T1', '5p'],
            ['D1', '9s'],
            ['D1', '8s'],
            ['chi', [0, 1], '2m', 3]  # 複雑なアクション
        ]
        
        compressed = result_processor._compress_actions(actions)
        
        # 圧縮結果の検証
        assert len(compressed) <= len(actions)
        # 複雑なアクションは圧縮されない
        assert ['chi', [0, 1], '2m', 3] in compressed
    
    def test_export_statistics(self, result_processor, temp_dir):
        """統計情報エクスポートのテスト"""
        output_path = os.path.join(temp_dir, 'game.json')
        statistics = {
            'total_frames': 1000,
            'detected_tiles': 2500,
            'processing_time': 45.6,
            'average_confidence': 0.85
        }
        
        # エクスポート実行
        result_processor.export_statistics(statistics, output_path)
        
        # 統計ファイルが作成されたことを確認
        stats_path = os.path.join(temp_dir, 'game.stats.json')
        assert os.path.exists(stats_path)
        
        # 内容を確認
        with open(stats_path, 'r', encoding='utf-8') as f:
            saved_stats = json.load(f)
        
        assert saved_stats == statistics
    
    def test_save_results_error_handling(self, result_processor):
        """エラーハンドリングのテスト"""
        # 無効なパスでの保存
        invalid_path = '/invalid/path/that/does/not/exist/result.json'
        game_data = Mock()
        game_data.to_tenhou_format.return_value = {'test': 'data'}
        
        # エラーが発生することを確認
        with pytest.raises(Exception):
            result_processor.save_results(game_data, invalid_path)
    
    def test_handle_game_data_without_to_tenhou_format(self, result_processor, temp_dir):
        """to_tenhou_formatメソッドを持たないゲームデータの処理テスト"""
        # to_tenhou_formatを持たないゲームデータ
        game_data = Mock(spec=[])  # spec=[]でメソッドを持たないモックを作成
        game_data.game_info = {'room_name': 'Test'}
        game_data.players = []
        game_data.rounds = []
        
        output_path = os.path.join(temp_dir, 'no_format_method.json')
        
        # FormatterのformatメソッドをモックSS
        with patch.object(result_processor.formatter, 'format_game_data') as mock_format:
            mock_format.return_value = {'formatted': 'data'}
            
            # 保存実行
            result_processor.save_results(game_data, output_path)
            
            # フォーマッターが呼び出されたことを確認
            mock_format.assert_called_once()
        
        # ファイルが作成されたことを確認
        assert os.path.exists(output_path)