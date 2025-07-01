"""
データ管理とバリデーションの統合テスト

データの入力、処理、バリデーション、出力の一連の流れをテスト
"""

import tempfile
from pathlib import Path

import pytest

from src.output.tenhou_game_data import (
    ActionType,
    CallType,
    GameType,
    TenhouAction,
    TenhouCall,
    TenhouGameData,
    TenhouGameRule,
    TenhouPlayerState,
    TenhouTile,
)
from src.output.tenhou_json_formatter import TenhouJsonFormatter
from src.training.dataset_manager import DatasetManager
from src.utils.config import ConfigManager
from src.utils.file_io import FileIOHelper
from src.validation.quality_validator import QualityValidator
from src.validation.tenhou_validator import TenhouValidator


class TestDataValidationIntegration:
    """データ管理とバリデーション統合テスト"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクト"""
        return ConfigManager()

    @pytest.fixture
    def sample_game_data(self):
        """サンプルゲームデータ"""
        game_data = TenhouGameData(
            players=[
                TenhouPlayerState(name="プレイヤー1", seat=0),
                TenhouPlayerState(name="プレイヤー2", seat=1),
                TenhouPlayerState(name="プレイヤー3", seat=2),
                TenhouPlayerState(name="プレイヤー4", seat=3),
            ],
            rule=TenhouGameRule(game_type=GameType.TONPUSEN),
        )

        # サンプルアクションを追加
        actions = [
            TenhouAction(
                action_type=ActionType.DRAW,
                player=0,
                tile=TenhouTile(tile_type="1m", tile_id=0),
                timestamp=1.0,
            ),
            TenhouAction(
                action_type=ActionType.DISCARD,
                player=0,
                tile=TenhouTile(tile_type="9p", tile_id=35),
                timestamp=2.0,
                is_tsumogiri=True,
            ),
            TenhouAction(
                action_type=ActionType.DRAW,
                player=1,
                tile=TenhouTile(tile_type="5s", tile_id=68),
                timestamp=3.0,
            ),
            TenhouAction(
                action_type=ActionType.CALL,
                player=1,
                call=TenhouCall(
                    call_type=CallType.PON,
                    tiles=[
                        TenhouTile(tile_type="5s", tile_id=68),
                        TenhouTile(tile_type="5s", tile_id=69),
                        TenhouTile(tile_type="5s", tile_id=70),
                    ],
                    from_player=0,
                ),
                timestamp=4.0,
            ),
        ]

        for action in actions:
            game_data.add_action(action)

        return game_data

    def test_data_format_conversion_flow(self, config_manager, sample_game_data):
        """データフォーマット変換フローのテスト"""
        # 1. TenhouGameData → JSON形式
        formatter = TenhouJsonFormatter()
        tenhou_json = formatter.format_game_data(sample_game_data, validate=False)

        assert tenhou_json is not None
        assert "title" in tenhou_json
        assert "log" in tenhou_json
        assert len(tenhou_json["log"]) == 4

        # 2. JSONの保存と読み込み
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "game_record.json"
            FileIOHelper.save_json(tenhou_json, json_path)

            # ファイルが作成されたことを確認
            assert json_path.exists()

            # 読み込み
            loaded_json = FileIOHelper.load_json(json_path)
            assert loaded_json == tenhou_json

    def test_validation_pipeline(self, config_manager, sample_game_data):
        """バリデーションパイプラインのテスト"""
        # 1. ゲームデータをJSON形式に変換
        formatter = TenhouJsonFormatter()
        tenhou_json = formatter.format_game_data(sample_game_data, validate=False)

        # 2. Tenhouバリデーター
        tenhou_validator = TenhouValidator()
        tenhou_result = tenhou_validator.validate_tenhou_json(tenhou_json)
        assert tenhou_result.is_valid

        # 3. 品質バリデーター
        quality_validator = QualityValidator(config_manager)
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "test_record.json"
            FileIOHelper.save_json(tenhou_json, json_path)

            quality_result = quality_validator.validate_record_file(json_path)
            assert quality_result["is_valid"]

    def test_dataset_integration(self, config_manager):
        """データセット管理との統合テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # データセットマネージャーを初期化
            dataset_manager = DatasetManager(base_dir=temp_dir)
            dataset_manager.initialize_database()

            # ビデオを登録
            video_id = dataset_manager.register_video("test_video.mp4", duration=120.0, fps=30.0)
            assert video_id is not None

            # フレームを追加
            frame_id = dataset_manager.add_frame(
                video_id=video_id, frame_number=100, timestamp=3.33, frame_path="frame_100.jpg"
            )
            assert frame_id is not None

            # アノテーションを追加
            annotation_id = dataset_manager.add_annotation(
                frame_id=frame_id,
                bbox=[100, 100, 50, 70],
                class_label="1m",
                confidence=0.95,
                annotator="test_user",
            )
            assert annotation_id is not None

            # データセットバージョンを作成
            version_id = dataset_manager.create_dataset_version(
                name="test_version", description="Integration test version"
            )
            assert version_id is not None

            # エクスポート
            export_dir = Path(temp_dir) / "export"
            dataset_manager.export_dataset_version(version_id, export_dir, format_type="yolo")
            assert export_dir.exists()

    def test_error_data_handling(self, config_manager):
        """エラーデータの処理テスト"""
        # 不正なゲームデータ
        invalid_game_data = TenhouGameData(
            players=[  # 3人しかいない（4人必要）
                TenhouPlayerState(name="P1", seat=0),
                TenhouPlayerState(name="P2", seat=1),
                TenhouPlayerState(name="P3", seat=2),
            ]
        )

        formatter = TenhouJsonFormatter()
        validator = TenhouValidator()

        # フォーマット変換（エラーでもJSONは生成される）
        json_data = formatter.format_game_data(invalid_game_data, validate=False)
        assert json_data is not None

        # バリデーション（エラーが検出される）
        result = validator.validate_tenhou_json(json_data)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_large_data_processing(self, config_manager):
        """大量データの処理テスト"""
        # 大きなゲームデータを作成
        large_game_data = TenhouGameData(
            players=[TenhouPlayerState(name=f"P{i}", seat=i) for i in range(4)]
        )

        # 1000アクションを追加
        for i in range(1000):
            player = i % 4
            if i % 2 == 0:
                action = TenhouAction(
                    action_type=ActionType.DRAW,
                    player=player,
                    tile=TenhouTile(tile_type="1m", tile_id=i % 136),
                    timestamp=float(i),
                )
            else:
                action = TenhouAction(
                    action_type=ActionType.DISCARD,
                    player=player,
                    tile=TenhouTile(tile_type="9p", tile_id=i % 136),
                    timestamp=float(i),
                )
            large_game_data.add_action(action)

        # 処理時間を計測
        import time

        start_time = time.time()

        # フォーマット変換
        formatter = TenhouJsonFormatter()
        json_data = formatter.format_game_data(large_game_data, validate=False)

        # バリデーション
        validator = TenhouValidator()
        validator.validate_tenhou_json(json_data)

        processing_time = time.time() - start_time

        # 処理が完了し、妥当な時間内であることを確認
        assert json_data is not None
        assert len(json_data["log"]) == 1000
        assert processing_time < 5.0  # 5秒以内

    def test_data_consistency_across_formats(self, config_manager, sample_game_data):
        """異なるフォーマット間でのデータ一貫性テスト"""
        # 1. 元のデータから統計を取得
        original_stats = sample_game_data.get_statistics()

        # 2. JSON形式に変換
        formatter = TenhouJsonFormatter()
        json_data = formatter.format_game_data(sample_game_data)

        # 3. JSONから再構築（仮想的なテスト）
        assert len(json_data["log"]) == original_stats["total_actions"]
        assert len(json_data["name"]) == 4

        # 4. 保存と読み込みでデータが保持されることを確認
        with tempfile.TemporaryDirectory() as temp_dir:
            # JSON保存
            json_path = Path(temp_dir) / "test.json"
            FileIOHelper.save_json(json_data, json_path, pretty=True)

            # YAML保存
            yaml_path = Path(temp_dir) / "test.yaml"
            FileIOHelper.save_yaml({"game_data": json_data}, yaml_path)

            # 読み込みと比較
            loaded_json = FileIOHelper.load_json(json_path)
            loaded_yaml = FileIOHelper.load_yaml(yaml_path)

            assert loaded_json == json_data
            assert loaded_yaml["game_data"] == json_data

    @pytest.mark.parametrize(
        "error_type,expected_valid",
        [
            ("missing_players", False),
            ("invalid_tile", False),
            ("negative_timestamp", False),
            ("valid_data", True),
        ],
    )
    def test_validation_scenarios(self, config_manager, error_type, expected_valid):
        """様々なバリデーションシナリオのテスト"""
        if error_type == "missing_players":
            game_data = TenhouGameData(players=[])
        elif error_type == "invalid_tile":
            game_data = TenhouGameData(
                players=[TenhouPlayerState(name=f"P{i}", seat=i) for i in range(4)]
            )
            game_data.add_action(
                TenhouAction(
                    action_type=ActionType.DRAW,
                    player=0,
                    tile=TenhouTile(tile_type="invalid", tile_id=999),  # 無効な牌
                    timestamp=1.0,
                )
            )
        elif error_type == "negative_timestamp":
            game_data = TenhouGameData(
                players=[TenhouPlayerState(name=f"P{i}", seat=i) for i in range(4)]
            )
            game_data.add_action(
                TenhouAction(
                    action_type=ActionType.DRAW,
                    player=0,
                    tile=TenhouTile(tile_type="1m", tile_id=0),
                    timestamp=-1.0,  # 負のタイムスタンプ
                )
            )
        else:  # valid_data
            game_data = TenhouGameData(
                players=[TenhouPlayerState(name=f"P{i}", seat=i) for i in range(4)]
            )
            game_data.add_action(
                TenhouAction(
                    action_type=ActionType.DRAW,
                    player=0,
                    tile=TenhouTile(tile_type="1m", tile_id=0),
                    timestamp=1.0,
                )
            )

        # バリデーション実行
        formatter = TenhouJsonFormatter()
        json_data = formatter.format_game_data(game_data, validate=False)

        validator = TenhouValidator()
        result = validator.validate_tenhou_json(json_data)

        assert result.is_valid == expected_valid
