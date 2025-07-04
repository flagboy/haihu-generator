"""
知識蒸留コンポーネント

教師モデルから生徒モデルへの知識転移を行う
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from ....utils.logger import LoggerMixin


class KnowledgeDistillationLoss(nn.Module if nn else object):
    """知識蒸留用の損失関数"""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.7, reduction: str = "mean"):
        """
        初期化

        Args:
            temperature: 蒸留温度（ソフトターゲットの滑らかさ）
            alpha: 蒸留損失の重み（1-alphaが通常の損失の重み）
            reduction: 損失の集約方法
        """
        if TORCH_AVAILABLE:
            super().__init__()

        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
        base_criterion: nn.Module,
    ) -> torch.Tensor:
        """
        知識蒸留損失を計算

        Args:
            student_outputs: 生徒モデルの出力
            teacher_outputs: 教師モデルの出力
            targets: 正解ラベル
            base_criterion: ベースとなる損失関数

        Returns:
            総合損失
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorchが利用できません")

        # 通常の損失（生徒の出力と正解ラベル）
        base_loss = base_criterion(student_outputs, targets)

        # 蒸留損失（生徒と教師の出力の差）
        # 温度でスケーリングしてソフトマックスを適用
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_outputs / self.temperature, dim=-1)

        # KLダイバージェンス
        distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction=self.reduction) * (
            self.temperature**2
        )

        # 総合損失
        total_loss = (1 - self.alpha) * base_loss + self.alpha * distillation_loss

        return total_loss


class DistillationTrainer(LoggerMixin):
    """知識蒸留トレーナー"""

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7,
    ):
        """
        初期化

        Args:
            teacher_model: 教師モデル
            student_model: 生徒モデル
            temperature: 蒸留温度
            alpha: 蒸留損失の重み
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_loss = KnowledgeDistillationLoss(temperature, alpha)

        # 教師モデルを評価モードに設定
        self.teacher_model.eval()

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        base_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> dict[str, float]:
        """
        1ステップの学習

        Args:
            inputs: 入力データ
            targets: 正解ラベル
            base_criterion: ベース損失関数
            optimizer: オプティマイザー
            device: デバイス

        Returns:
            メトリクス
        """
        # デバイスに転送
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 教師モデルの出力を取得（勾配計算なし）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        # 生徒モデルの出力を取得
        student_outputs = self.student_model(inputs)

        # 知識蒸留損失を計算
        loss = self.distillation_loss(student_outputs, teacher_outputs, targets, base_criterion)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # メトリクスを計算
        with torch.no_grad():
            # 精度
            _, predicted = torch.max(student_outputs, 1)
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0)

            # 教師との一致率
            _, teacher_predicted = torch.max(teacher_outputs, 1)
            agreement = (predicted == teacher_predicted).sum().item() / targets.size(0)

        return {"loss": loss.item(), "accuracy": accuracy, "teacher_agreement": agreement}

    def extract_features(
        self, model: nn.Module, layer_name: str, inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        中間層の特徴を抽出

        Args:
            model: モデル
            layer_name: レイヤー名
            inputs: 入力データ

        Returns:
            特徴マップ
        """
        features = {}

        def hook_fn(module, input, output):
            features["output"] = output

        # フックを登録
        layer = dict(model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)

        # フォワードパス
        with torch.no_grad():
            _ = model(inputs)

        # フックを削除
        hook.remove()

        return features.get("output")

    def compute_feature_distillation_loss(
        self, student_features: torch.Tensor, teacher_features: torch.Tensor, loss_type: str = "mse"
    ) -> torch.Tensor:
        """
        特徴量レベルの蒸留損失を計算

        Args:
            student_features: 生徒モデルの特徴
            teacher_features: 教師モデルの特徴
            loss_type: 損失タイプ（"mse" or "cosine"）

        Returns:
            特徴蒸留損失
        """
        if loss_type == "mse":
            # 平均二乗誤差
            return F.mse_loss(student_features, teacher_features)
        elif loss_type == "cosine":
            # コサイン類似度損失
            student_norm = F.normalize(student_features, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
            return 1 - (student_norm * teacher_norm).sum(dim=-1).mean()
        else:
            raise ValueError(f"未対応の損失タイプ: {loss_type}")


class AdaptiveDistillation(LoggerMixin):
    """適応的知識蒸留"""

    def __init__(
        self,
        initial_temperature: float = 3.0,
        initial_alpha: float = 0.7,
        temperature_schedule: str = "linear",
        alpha_schedule: str = "constant",
    ):
        """
        初期化

        Args:
            initial_temperature: 初期温度
            initial_alpha: 初期アルファ
            temperature_schedule: 温度スケジュール
            alpha_schedule: アルファスケジュール
        """
        self.initial_temperature = initial_temperature
        self.initial_alpha = initial_alpha
        self.temperature_schedule = temperature_schedule
        self.alpha_schedule = alpha_schedule

        self.current_temperature = initial_temperature
        self.current_alpha = initial_alpha
        self.step_count = 0

    def update_parameters(self, epoch: int, total_epochs: int):
        """
        パラメータを更新

        Args:
            epoch: 現在のエポック
            total_epochs: 総エポック数
        """
        progress = epoch / total_epochs

        # 温度の更新
        if self.temperature_schedule == "linear":
            # 線形に減少
            self.current_temperature = self.initial_temperature * (1 - 0.9 * progress)
        elif self.temperature_schedule == "exponential":
            # 指数的に減少
            self.current_temperature = self.initial_temperature * (0.1**progress)
        elif self.temperature_schedule == "cosine":
            # コサインアニーリング
            import math

            self.current_temperature = self.initial_temperature * (
                0.5 * (1 + math.cos(math.pi * progress))
            )

        # アルファの更新
        if self.alpha_schedule == "linear":
            # 徐々に通常の損失の重みを増やす
            self.current_alpha = self.initial_alpha * (1 - progress)
        elif self.alpha_schedule == "step" and progress > 0.5:
            # ステップ的に変更
            self.current_alpha = self.initial_alpha * 0.5

        self.logger.info(
            f"蒸留パラメータ更新: "
            f"温度={self.current_temperature:.2f}, "
            f"アルファ={self.current_alpha:.2f}"
        )

    def get_current_parameters(self) -> dict[str, float]:
        """現在のパラメータを取得"""
        return {
            "temperature": self.current_temperature,
            "alpha": self.current_alpha,
            "step_count": self.step_count,
        }
