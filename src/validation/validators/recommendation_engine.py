"""
推奨事項生成エンジン
"""

from typing import Any

from ..validators.base_validator import ValidationIssue


class RecommendationEngine:
    """推奨事項生成エンジン"""

    def __init__(self, validation_config: dict[str, Any]):
        """
        初期化

        Args:
            validation_config: 検証設定
        """
        self.validation_config = validation_config
        self.thresholds = validation_config.get("thresholds", {})

    def generate_recommendations(
        self, issues: list[ValidationIssue], category_scores: dict[str, float], overall_score: float
    ) -> list[str]:
        """
        推奨事項を生成

        Args:
            issues: 検証問題リスト
            category_scores: カテゴリ別スコア
            overall_score: 総合スコア

        Returns:
            推奨事項リスト
        """
        recommendations = []

        # エラーに基づく推奨
        error_recommendations = self._generate_error_recommendations(issues)
        recommendations.extend(error_recommendations)

        # カテゴリ別推奨
        category_recommendations = self._generate_category_recommendations(category_scores)
        recommendations.extend(category_recommendations)

        # 総合評価に基づく推奨
        overall_recommendations = self._generate_overall_recommendations(overall_score)
        recommendations.extend(overall_recommendations)

        # 重複を除去
        return list(dict.fromkeys(recommendations))

    def _generate_error_recommendations(self, issues: list[ValidationIssue]) -> list[str]:
        """エラーベースの推奨事項を生成"""
        recommendations = []

        # エラー数を集計
        error_count = len([issue for issue in issues if issue.severity == "error"])
        warning_count = len([issue for issue in issues if issue.severity == "warning"])

        if error_count > 0:
            recommendations.append(f"{error_count}個のエラーを修正してください")

            # カテゴリ別エラーの分析
            error_categories = {}
            for issue in issues:
                if issue.severity == "error":
                    category = issue.category.value
                    error_categories[category] = error_categories.get(category, 0) + 1

            # 最も多いエラーカテゴリに対する推奨
            if error_categories:
                max_category = max(error_categories.items(), key=lambda x: x[1])
                recommendations.append(
                    f"特に{max_category[0]}関連のエラー({max_category[1]}個)に注意してください"
                )

        if warning_count > 0:
            recommendations.append(f"{warning_count}個の警告を確認してください")

        # 具体的な修正提案
        for issue in issues[:3]:  # 最初の3つの問題に対する提案
            if issue.suggestion:
                recommendations.append(f"修正提案: {issue.suggestion}")

        return recommendations

    def _generate_category_recommendations(self, category_scores: dict[str, float]) -> list[str]:
        """カテゴリ別の推奨事項を生成"""
        recommendations = []

        pass_score = self.thresholds.get("pass_score", 70.0)
        good_score = self.thresholds.get("good_score", 85.0)

        category_names = {
            "structure": "構造",
            "content": "内容",
            "logic": "論理",
            "consistency": "一貫性",
        }

        for category, score in category_scores.items():
            if score < pass_score:
                recommendations.append(
                    f"{category_names.get(category, category)}の品質を改善してください（現在: {score:.1f}）"
                )

                # カテゴリ別の具体的なアドバイス
                if category == "structure" and score < 50:
                    recommendations.append("基本的なJSON構造を確認してください")
                elif category == "content" and score < 50:
                    recommendations.append("牌名やプレイヤー情報を確認してください")
                elif category == "logic" and score < 50:
                    recommendations.append("ゲームルールに従った進行になっているか確認してください")
                elif category == "consistency" and score < 50:
                    recommendations.append("データの一貫性を確認してください")

            elif score < good_score:
                recommendations.append(
                    f"{category_names.get(category, category)}の品質向上を検討してください（現在: {score:.1f}）"
                )

        return recommendations

    def _generate_overall_recommendations(self, overall_score: float) -> list[str]:
        """総合評価に基づく推奨事項を生成"""
        recommendations = []

        pass_score = self.thresholds.get("pass_score", 70.0)
        good_score = self.thresholds.get("good_score", 85.0)
        excellent_score = self.thresholds.get("excellent_score", 95.0)

        if overall_score >= excellent_score:
            recommendations.append("優秀な品質です")
            recommendations.append("現在の品質を維持してください")
        elif overall_score >= good_score:
            recommendations.append("良好な品質です")
            recommendations.append(f"優秀レベル（{excellent_score}点以上）を目指しましょう")
        elif overall_score >= pass_score:
            recommendations.append("合格レベルの品質です")
            recommendations.append("さらなる品質向上を検討してください")
        else:
            recommendations.append("品質が基準を満たしていません")
            recommendations.append(f"合格レベル（{pass_score}点以上）を目指して改善してください")

            # 改善優先度の提案
            if overall_score < 30:
                recommendations.append("まず基本的な構造の問題から修正を始めてください")
            elif overall_score < 50:
                recommendations.append("エラーの修正を優先的に行ってください")
            else:
                recommendations.append("警告事項の確認と修正を行ってください")

        return recommendations

    def get_improvement_priority(self, category_scores: dict[str, float]) -> list[str]:
        """改善優先度を取得"""
        # スコアが低い順にソート
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1])

        priorities = []
        for category, score in sorted_categories:
            if score < self.thresholds.get("pass_score", 70.0):
                priorities.append(category)

        return priorities
