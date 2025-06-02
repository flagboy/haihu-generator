"""
フェーズ3（ゲーム状態追跡・ロジック実装）のデモスクリプト
"""

import time
from typing import Any

from src.game.player import PlayerPosition
from src.pipeline.game_pipeline import GamePipeline
from src.utils.logger import get_logger


def create_sample_frame_data(frame_number: int, scenario: str) -> dict[str, Any]:
    """
    サンプルフレームデータを作成

    Args:
        frame_number: フレーム番号
        scenario: シナリオ名

    Returns:
        Dict[str, Any]: フレームデータ
    """
    base_data = {
        "frame_number": frame_number,
        "timestamp": time.time(),
        "confidence_scores": {"detection": 0.85, "classification": 0.80},
    }

    if scenario == "initial_deal":
        # 配牌シナリオ
        base_data.update(
            {
                "player_hands": {
                    "0": [
                        "1m",
                        "2m",
                        "3m",
                        "4m",
                        "5m",
                        "6m",
                        "7m",
                        "8m",
                        "9m",
                        "1p",
                        "2p",
                        "3p",
                        "4p",
                    ],
                    "1": [
                        "5p",
                        "6p",
                        "7p",
                        "8p",
                        "9p",
                        "1s",
                        "2s",
                        "3s",
                        "4s",
                        "5s",
                        "6s",
                        "7s",
                        "8s",
                    ],
                    "2": [
                        "9s",
                        "東",
                        "南",
                        "西",
                        "北",
                        "白",
                        "發",
                        "中",
                        "1m",
                        "2m",
                        "3m",
                        "4m",
                        "5m",
                    ],
                    "3": [
                        "6m",
                        "7m",
                        "8m",
                        "9m",
                        "1p",
                        "2p",
                        "3p",
                        "4p",
                        "5p",
                        "6p",
                        "7p",
                        "8p",
                        "9p",
                    ],
                },
                "discarded_tiles": {"0": [], "1": [], "2": [], "3": []},
            }
        )

    elif scenario == "first_discard":
        # 最初の打牌シナリオ
        base_data.update(
            {
                "player_hands": {
                    "0": [
                        "1m",
                        "2m",
                        "3m",
                        "4m",
                        "5m",
                        "6m",
                        "7m",
                        "8m",
                        "9m",
                        "1p",
                        "2p",
                        "3p",
                    ],  # 1枚減
                    "1": [
                        "5p",
                        "6p",
                        "7p",
                        "8p",
                        "9p",
                        "1s",
                        "2s",
                        "3s",
                        "4s",
                        "5s",
                        "6s",
                        "7s",
                        "8s",
                    ],
                    "2": [
                        "9s",
                        "東",
                        "南",
                        "西",
                        "北",
                        "白",
                        "發",
                        "中",
                        "1m",
                        "2m",
                        "3m",
                        "4m",
                        "5m",
                    ],
                    "3": [
                        "6m",
                        "7m",
                        "8m",
                        "9m",
                        "1p",
                        "2p",
                        "3p",
                        "4p",
                        "5p",
                        "6p",
                        "7p",
                        "8p",
                        "9p",
                    ],
                },
                "discarded_tiles": {
                    "0": ["4p"],  # 打牌
                    "1": [],
                    "2": [],
                    "3": [],
                },
            }
        )

    elif scenario == "draw_and_discard":
        # ツモ切りシナリオ
        base_data.update(
            {
                "player_hands": {
                    "0": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"],
                    "1": [
                        "5p",
                        "6p",
                        "7p",
                        "8p",
                        "9p",
                        "1s",
                        "2s",
                        "3s",
                        "4s",
                        "5s",
                        "6s",
                        "7s",
                    ],  # 1枚減
                    "2": [
                        "9s",
                        "東",
                        "南",
                        "西",
                        "北",
                        "白",
                        "發",
                        "中",
                        "1m",
                        "2m",
                        "3m",
                        "4m",
                        "5m",
                    ],
                    "3": [
                        "6m",
                        "7m",
                        "8m",
                        "9m",
                        "1p",
                        "2p",
                        "3p",
                        "4p",
                        "5p",
                        "6p",
                        "7p",
                        "8p",
                        "9p",
                    ],
                },
                "discarded_tiles": {
                    "0": ["4p"],
                    "1": ["8s"],  # 新しい打牌
                    "2": [],
                    "3": [],
                },
            }
        )

    elif scenario == "multiple_discards":
        # 複数打牌シナリオ
        base_data.update(
            {
                "player_hands": {
                    "0": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"],
                    "1": ["5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s"],
                    "2": [
                        "9s",
                        "東",
                        "南",
                        "西",
                        "北",
                        "白",
                        "發",
                        "中",
                        "1m",
                        "2m",
                        "3m",
                        "4m",
                    ],  # 1枚減
                    "3": [
                        "6m",
                        "7m",
                        "8m",
                        "9m",
                        "1p",
                        "2p",
                        "3p",
                        "4p",
                        "5p",
                        "6p",
                        "7p",
                        "8p",
                    ],  # 1枚減
                },
                "discarded_tiles": {
                    "0": ["4p"],
                    "1": ["8s"],
                    "2": ["5m"],  # 新しい打牌
                    "3": ["9p"],  # 新しい打牌
                },
            }
        )

    return base_data


def run_game_simulation():
    """ゲームシミュレーションを実行"""
    logger = get_logger(__name__)
    logger.info("=== フェーズ3 ゲーム状態追跡デモ開始 ===")

    # パイプラインを初期化
    pipeline = GamePipeline("demo_game")

    # プレイヤー名を設定
    player_names = {
        PlayerPosition.EAST: "東家",
        PlayerPosition.SOUTH: "南家",
        PlayerPosition.WEST: "西家",
        PlayerPosition.NORTH: "北家",
    }

    # ゲームを初期化
    success = pipeline.initialize_game(player_names)
    if not success:
        logger.error("ゲーム初期化に失敗しました")
        return

    logger.info("ゲーム初期化完了")

    # 新しい局を開始
    pipeline.start_new_round(1, "東1局", PlayerPosition.EAST)
    logger.info("東1局開始")

    # シミュレーションシナリオ
    scenarios = [
        ("initial_deal", "配牌"),
        ("first_discard", "最初の打牌"),
        ("draw_and_discard", "ツモ切り"),
        ("multiple_discards", "複数プレイヤーの打牌"),
    ]

    frame_number = 1

    for scenario, description in scenarios:
        logger.info(f"\n--- {description} シナリオ ---")

        # フレームデータを作成
        frame_data = create_sample_frame_data(frame_number, scenario)

        # フレームを処理
        result = pipeline.process_frame(frame_data)

        # 結果を表示
        logger.info(f"フレーム {frame_number} 処理結果:")
        logger.info(f"  成功: {result.success}")
        logger.info(f"  検出行動数: {result.actions_detected}")
        logger.info(f"  信頼度: {result.confidence:.2f}")
        logger.info(f"  処理時間: {result.processing_time:.3f}秒")

        if result.errors:
            logger.warning(f"  エラー: {result.errors}")

        # ゲーム状態を表示
        game_summary = pipeline.game_state.get_game_summary()
        logger.info(f"  ゲーム状態: {game_summary['phase']}")
        logger.info(f"  フレーム数: {game_summary['frame_count']}")

        frame_number += 1
        time.sleep(0.1)  # 少し待機

    # 統計情報を表示
    logger.info("\n=== パイプライン統計 ===")
    stats = pipeline.get_pipeline_statistics()

    logger.info(f"総フレーム数: {stats['total_frames']}")
    logger.info(f"成功フレーム数: {stats['successful_frames']}")
    logger.info(f"失敗フレーム数: {stats['failed_frames']}")
    logger.info(f"成功率: {stats['success_rate']:.2%}")
    logger.info(f"平均処理時間: {stats['average_frame_time']:.3f}秒")

    # 追跡統計を表示
    tracking_stats = stats["tracking_statistics"]
    logger.info(f"追跡状態: {tracking_stats['tracking_state']}")
    logger.info(f"現在の信頼度: {tracking_stats['current_confidence']:.2f}")
    logger.info(f"矛盾検出数: {tracking_stats['inconsistency_count']}")

    # 局を完了
    result_data = {
        "result_type": "ryukyoku",  # 流局
        "reason": "demo_end",
    }
    pipeline.complete_current_round(result_data)
    logger.info("局完了")

    # 牌譜をエクスポート
    logger.info("\n=== 牌譜エクスポート ===")

    # MJSCORE形式
    mjscore_data = pipeline.export_game_record("mjscore")
    logger.info("MJSCORE形式牌譜:")
    logger.info(mjscore_data[:500] + "..." if len(mjscore_data) > 500 else mjscore_data)

    # 天鳳形式
    tenhou_data = pipeline.export_game_record("tenhou")
    logger.info("\n天鳳形式牌譜:")
    logger.info(tenhou_data[:300] + "..." if len(tenhou_data) > 300 else tenhou_data)

    logger.info("\n=== デモ完了 ===")


def run_tracking_demo():
    """状態追跡機能のデモ"""
    logger = get_logger(__name__)
    logger.info("\n=== 状態追跡機能デモ ===")

    from src.tracking.action_detector import ActionDetector
    from src.tracking.change_analyzer import ChangeAnalyzer

    # 行動検出器のテスト
    detector = ActionDetector()

    # サンプルフレームデータで行動検出
    frame1 = create_sample_frame_data(1, "initial_deal")
    frame2 = create_sample_frame_data(2, "first_discard")

    # 最初のフレーム
    result1 = detector.detect_actions(frame1, 1)
    logger.info(f"フレーム1 検出結果: {len(result1.actions)}個の行動")

    # 2番目のフレーム
    result2 = detector.detect_actions(frame2, 2)
    logger.info(f"フレーム2 検出結果: {len(result2.actions)}個の行動")

    for action in result2.actions:
        logger.info(
            f"  行動: {action.action_type.value}, "
            f"プレイヤー: {action.player.name}, 牌: {action.tile}"
        )

    # 変化分析器のテスト
    analyzer = ChangeAnalyzer()

    changes = analyzer.analyze_frame_changes(frame1, frame2, 2)
    logger.info(f"変化分析結果: {len(changes)}個の変化")

    for change in changes:
        logger.info(
            f"  変化: {change.change_type.value}, 場所: {change.location.value}, 牌: {change.tiles}"
        )

    # 統計情報
    change_stats = analyzer.get_change_statistics()
    logger.info(f"変化統計: {change_stats}")


def main():
    """メイン関数"""
    print("麻雀牌譜作成システム フェーズ3 デモ")
    print("=" * 50)

    try:
        # ゲームシミュレーション
        run_game_simulation()

        # 状態追跡デモ
        run_tracking_demo()

        print("\nデモが正常に完了しました！")

    except Exception as e:
        print(f"デモ実行中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
