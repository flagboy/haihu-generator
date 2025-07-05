"""
点数読み取り機能のテスト
"""

import numpy as np
import pytest

from src.detection import PlayerScore, ScoreReader


class TestScoreReader:
    """ScoreReaderのテスト"""

    @pytest.fixture
    def reader(self):
        """テスト用読み取り器"""
        config = {
            "ocr_lang": "eng",
            "preprocessing": True,
            "min_confidence": 0.6,
            "score_regions": {
                "east": {"x": 0.85, "y": 0.5, "w": 0.1, "h": 0.05},
                "south": {"x": 0.45, "y": 0.85, "w": 0.1, "h": 0.05},
                "west": {"x": 0.05, "y": 0.5, "w": 0.1, "h": 0.05},
                "north": {"x": 0.45, "y": 0.05, "w": 0.1, "h": 0.05},
            },
        }
        return ScoreReader(config)

    @pytest.fixture
    def sample_frame(self):
        """テスト用サンプルフレーム"""
        # 1920x1080のフレームを作成
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 50  # グレー背景

        # 各位置に白い領域を配置（点数表示を模擬）
        # 東
        frame[540 - 27 : 540 + 27, 1632 - 96 : 1632 + 96] = 255
        # 南
        frame[918 - 27 : 918 + 27, 864 - 96 : 864 + 96] = 255
        # 西
        frame[540 - 27 : 540 + 27, 96 - 48 : 96 + 48] = 255
        # 北
        frame[54 - 27 : 54 + 27, 864 - 96 : 864 + 96] = 255

        return frame

    def test_initialization(self, reader):
        """初期化のテスト"""
        assert reader.ocr_lang == "eng"
        assert reader.preprocessing is True
        assert reader.min_confidence == 0.6
        assert len(reader.score_regions) == 4
        assert reader.prev_scores is None

    def test_read_scores_basic(self, reader, sample_frame):
        """基本的な点数読み取りテスト"""
        result = reader.read_scores(sample_frame, frame_number=100, timestamp=3.33)

        assert result.frame_number == 100
        assert result.timestamp == 3.33
        assert isinstance(result.scores, list)
        assert len(result.scores) <= 4

        # 各スコアの検証
        for score in result.scores:
            assert isinstance(score, PlayerScore)
            assert score.player_position in ["east", "south", "west", "north"]
            assert isinstance(score.score, int)
            assert 0 <= score.confidence <= 1.0
            assert len(score.bbox) == 4

    def test_get_score_by_position(self, reader, sample_frame):
        """位置指定での点数取得テスト"""
        result = reader.read_scores(sample_frame, 0, 0.0)

        # 各位置の点数を取得
        east_score = result.get_score_by_position("east")
        if east_score:
            assert east_score.player_position == "east"

        # 存在しない位置
        invalid_score = result.get_score_by_position("invalid")
        assert invalid_score is None

    def test_is_valid_result(self, reader):
        """結果の妥当性チェックテスト"""
        # 有効な結果（4人で合計100,000点）
        from src.detection.score_reader import ScoreReadingResult

        valid_result = ScoreReadingResult(
            frame_number=0,
            timestamp=0.0,
            scores=[
                PlayerScore("east", 25000, 0.8, (0, 0, 100, 50)),
                PlayerScore("south", 25000, 0.8, (0, 0, 100, 50)),
                PlayerScore("west", 25000, 0.8, (0, 0, 100, 50)),
                PlayerScore("north", 25000, 0.8, (0, 0, 100, 50)),
            ],
            total_confidence=0.8,
        )
        assert valid_result.is_valid() is True

        # 無効な結果（3人しかいない）
        invalid_result = ScoreReadingResult(
            frame_number=0,
            timestamp=0.0,
            scores=[
                PlayerScore("east", 30000, 0.8, (0, 0, 100, 50)),
                PlayerScore("south", 30000, 0.8, (0, 0, 100, 50)),
                PlayerScore("west", 40000, 0.8, (0, 0, 100, 50)),
            ],
            total_confidence=0.8,
        )
        assert invalid_result.is_valid() is False

    def test_preprocess_for_ocr(self, reader):
        """OCR前処理のテスト"""
        # カラー画像
        color_image = np.ones((50, 100, 3), dtype=np.uint8) * 128
        processed = reader._preprocess_for_ocr(color_image)

        assert len(processed.shape) == 2  # グレースケール
        assert processed.shape[0] > 50  # リサイズされている
        assert processed.shape[1] > 100

        # グレースケール画像
        gray_image = np.ones((50, 100), dtype=np.uint8) * 128
        processed = reader._preprocess_for_ocr(gray_image)
        assert len(processed.shape) == 2

    def test_extract_score_from_text(self, reader):
        """テキストからの点数抽出テスト"""
        # 正常なケース
        assert reader._extract_score_from_text("25,000") == 25000
        assert reader._extract_score_from_text("25000") == 25000
        assert reader._extract_score_from_text("Score: 12,300") == 12300

        # 100の倍数に丸める
        assert reader._extract_score_from_text("25,050") == 25000
        assert reader._extract_score_from_text("25,099") == 25000

        # 無効なケース
        assert reader._extract_score_from_text("abc") is None
        assert reader._extract_score_from_text("") is None

    def test_is_valid_score(self, reader):
        """点数の妥当性チェックテスト"""
        # 有効な点数
        assert reader._is_valid_score(25000) is True
        assert reader._is_valid_score(0) is True
        assert reader._is_valid_score(100) is True

        # 無効な点数
        assert reader._is_valid_score(25050) is False  # 100の倍数でない
        assert reader._is_valid_score(-1000) is False  # 負の値
        assert reader._is_valid_score(600000) is False  # 範囲外

    def test_detect_score_changes(self, reader, sample_frame):
        """点数変化検出のテスト"""
        # 初回読み取り
        reader.read_scores(sample_frame, 0, 0.0)

        # 2回目の読み取り（変化なし）
        reader.read_scores(sample_frame, 1, 0.033)

        # prev_scoresが更新されていることを確認
        assert reader.prev_scores is not None
