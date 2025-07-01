# Claude Code Hooks

このディレクトリには、Claude Codeで使用するGit フックが含まれています。

## pre-commit フック

`pre-commit`ファイルは、コミット前に自動的に実行されるスクリプトです。

### 機能

1. **コードフォーマット（Ruff）**
   - ステージされたPythonファイルのフォーマットをチェック
   - 問題があれば自動修正してステージに追加

2. **Lintチェック（Ruff）**
   - コード品質の問題を検出
   - 自動修正可能な問題は修正してステージに追加
   - 修正できない問題がある場合はコミットを中止

3. **型チェック（mypy）**
   - 型の整合性をチェック
   - 警告のみ（コミットは中止しない）

4. **関連テストの実行**
   - 変更されたファイルに関連するテストを自動実行
   - テストが失敗した場合はコミットを中止

5. **既存のpre-commitとの統合**
   - `.pre-commit-config.yaml`が存在する場合は追加で実行

### 使用方法

#### 方法1: プロジェクトローカルでの使用

```bash
# フックを実行可能にする（既に設定済み）
chmod +x .claude/hooks/pre-commit

# Claude Codeがこのフックを使用するよう設定
# （Claude Codeの設定に依存）
```

#### 方法2: グローバルhooksディレクトリへのリンク

```bash
# ユーザーのClaude Code hooksディレクトリにリンクを作成
ln -sf $(pwd)/.claude/hooks/pre-commit ~/.claude/hooks/haihu-generator-pre-commit
```

### 手動実行

フックを手動でテストする場合：

```bash
# 直接実行
./.claude/hooks/pre-commit

# 特定のファイルをステージしてテスト
git add src/some_file.py
./.claude/hooks/pre-commit
```

### トラブルシューティング

#### uvが見つからない

```bash
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### テストがタイムアウトする

フック内のpytestコマンドに`--timeout`オプションを追加：

```bash
uv run pytest $EXISTING_TESTS -x --tb=short -q --timeout=300
```

#### 特定のチェックをスキップしたい

環境変数で制御：

```bash
# 型チェックをスキップ
SKIP_MYPY=1 git commit -m "message"

# テストをスキップ
SKIP_TESTS=1 git commit -m "message"
```

### カスタマイズ

プロジェクトの要件に応じて、以下の部分をカスタマイズできます：

1. **ファイルサイズ制限**（26行目）
   ```bash
   MAX_FILE_SIZE=$((2 * 1024 * 1024 * 1024))  # 2GB
   ```

2. **テストのタイムアウト**
   ```bash
   uv run pytest $EXISTING_TESTS -x --tb=short -q --timeout=300
   ```

3. **追加のチェック**
   - セキュリティチェック（bandit）
   - ドキュメントチェック（pydocstyle）
   - その他のカスタムチェック

### 無効化

一時的にフックを無効化する場合：

```bash
# フックをスキップしてコミット
git commit --no-verify -m "message"
```

恒久的に無効化する場合：

```bash
# フックファイルを削除または名前変更
mv .claude/hooks/pre-commit .claude/hooks/pre-commit.disabled
```
