"""
フェーズ3の簡単な動作確認デモ
"""

import sys
import os
import time

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tile_definitions():
    """牌定義のテスト"""
    print("=== 牌定義テスト ===")
    
    from src.utils.tile_definitions import TileDefinitions, TileType
    
    tile_def = TileDefinitions()
    
    # 基本機能テスト
    print(f"全牌数: {tile_def.get_tile_count()}")
    print(f"萬子牌: {tile_def.get_tiles_by_type(TileType.MANZU)}")
    print(f"字牌: {tile_def.get_tiles_by_type(TileType.JIHAI)}")
    
    # 牌の判定テスト
    test_tiles = ['1m', '9s', '東', '5mr', 'invalid']
    for tile in test_tiles:
        print(f"{tile}: 有効={tile_def.is_valid_tile(tile)}, "
              f"数牌={tile_def.is_number_tile(tile)}, "
              f"字牌={tile_def.is_honor_tile(tile)}, "
              f"么九={tile_def.is_terminal_tile(tile)}")
    
    print("牌定義テスト完了\n")


def test_player():
    """プレイヤークラスのテスト"""
    print("=== プレイヤーテスト ===")
    
    from src.game.player import Player, PlayerPosition
    
    player = Player(PlayerPosition.EAST, "テストプレイヤー")
    
    # 手牌操作テスト
    tiles_to_add = ['1m', '2m', '3m', '4m', '5m']
    for tile in tiles_to_add:
        player.add_tile_to_hand(tile)
    
    print(f"プレイヤー: {player}")
    print(f"手牌: {player.get_sorted_hand()}")
    print(f"手牌枚数: {player.get_hand_size()}")
    
    # 打牌テスト
    discard_result = player.discard_tile('3m')
    print(f"3m打牌結果: {discard_result}")
    print(f"捨て牌: {player.state.discarded_tiles}")
    print(f"手牌: {player.get_sorted_hand()}")
    
    print("プレイヤーテスト完了\n")


def test_table():
    """テーブルクラスのテスト"""
    print("=== テーブルテスト ===")
    
    from src.game.table import Table, GameType, Wind
    
    table = Table(GameType.HANCHAN)
    
    print(f"テーブル: {table}")
    print(f"現在の局: {table.get_current_round_name()}")
    
    # 配牌テスト
    hands = table.shuffle_and_deal()
    print(f"配牌結果:")
    for pos, tiles in hands.items():
        print(f"  {pos}: {len(tiles)}枚")
    
    print(f"残り牌数: {table.state.remaining_tiles}")
    print(f"ドラ表示牌: {table.state.dora_indicators}")
    print(f"ドラ牌: {table.get_dora_tiles()}")
    
    print("テーブルテスト完了\n")


def test_turn_management():
    """ターン管理のテスト"""
    print("=== ターン管理テスト ===")
    
    from src.game.turn import Turn, Action, ActionType
    from src.game.player import PlayerPosition
    
    turn_manager = Turn()
    
    # ターン開始
    turn_state = turn_manager.start_new_turn(PlayerPosition.EAST)
    print(f"ターン開始: {turn_state.turn_number}, プレイヤー: {turn_state.current_player.name}")
    
    # 行動追加
    actions = [
        Action(ActionType.DRAW, PlayerPosition.EAST, tile='1m'),
        Action(ActionType.DISCARD, PlayerPosition.EAST, tile='9p')
    ]
    
    for action in actions:
        turn_manager.add_action(action)
        print(f"行動追加: {action.action_type.value}, 牌: {action.tile}")
    
    # 統計取得
    stats = turn_manager.get_statistics()
    print(f"ターン統計: {stats}")
    
    print("ターン管理テスト完了\n")


def test_action_detector():
    """行動検出のテスト"""
    print("=== 行動検出テスト ===")
    
    from src.tracking.action_detector import ActionDetector
    
    detector = ActionDetector()
    
    # サンプルフレームデータ
    frame1_data = {
        'frame_number': 1,
        'player_hands': {
            '0': ['1m', '2m', '3m', '4m', '5m']
        },
        'discarded_tiles': {
            '0': []
        }
    }
    
    frame2_data = {
        'frame_number': 2,
        'player_hands': {
            '0': ['1m', '2m', '3m', '4m']  # 1枚減
        },
        'discarded_tiles': {
            '0': ['5m']  # 打牌
        }
    }
    
    # 行動検出
    result1 = detector.detect_actions(frame1_data, 1)
    print(f"フレーム1: {len(result1.actions)}個の行動検出")
    
    result2 = detector.detect_actions(frame2_data, 2)
    print(f"フレーム2: {len(result2.actions)}個の行動検出")
    
    for action in result2.actions:
        print(f"  行動: {action.action_type.value}, プレイヤー: {action.player.name}, 牌: {action.tile}")
    
    print("行動検出テスト完了\n")


def test_game_state():
    """ゲーム状態のテスト"""
    print("=== ゲーム状態テスト ===")
    
    from src.game.game_state import GameState, GamePhase
    
    game_state = GameState()
    
    print(f"初期状態: {game_state}")
    print(f"フェーズ: {game_state.phase.value}")
    
    # フレームデータ更新テスト
    frame_data = {
        'frame_number': 1,
        'timestamp': time.time(),
        'player_hands': {
            '0': ['1m', '2m', '3m'],
            '1': ['4p', '5p', '6p']
        }
    }
    
    result = game_state.update_from_frame_detection(frame_data)
    print(f"フレーム更新結果: {result}")
    print(f"フレーム履歴: {len(game_state.frame_history)}個")
    
    # ゲーム概要取得
    summary = game_state.get_game_summary()
    print(f"ゲーム概要: フェーズ={summary['phase']}, フレーム数={summary['frame_count']}")
    
    print("ゲーム状態テスト完了\n")


def main():
    """メイン関数"""
    print("麻雀牌譜作成システム フェーズ3 簡単動作確認")
    print("=" * 60)
    
    try:
        test_tile_definitions()
        test_player()
        test_table()
        test_turn_management()
        test_action_detector()
        test_game_state()
        
        print("=" * 60)
        print("すべてのテストが正常に完了しました！")
        print("\nフェーズ3で実装された主要機能:")
        print("✓ ゲーム状態管理（GameState, Player, Table, Turn）")
        print("✓ 状態追跡システム（ActionDetector, StateTracker, ChangeAnalyzer）")
        print("✓ 履歴管理（HistoryManager）")
        print("✓ データ統合パイプライン（GamePipeline）")
        print("✓ 動画解析結果からの牌譜生成機能")
        
    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()