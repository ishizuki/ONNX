# 🧠 ONNX Text Classifier

このプロジェクトは、**テキスト分類モデルを ONNX 形式に変換し、ONNX Runtime を使って高速・軽量な推論**を実現するサンプルです。主に以下のことが行えます：

- `scikit-learn` でテキスト分類モデルを作成
- `skl2onnx` で ONNX フォーマットに変換
- `onnxruntime` で ONNX モデルの推論実行

---

## 📚 対象読者

- ONNX に興味がある Python エンジニア
- scikit-learn のモデルを軽量・高速にデプロイしたい方
- モバイルやエッジ向けの機械学習推論を検討している方

---

## 📦 使用ライブラリ

| ライブラリ名        | バージョン     | 役割                                 |
|---------------------|----------------|--------------------------------------|
| `scikit-learn`      | 1.3.2          | 機械学習モデル（テキスト分類）作成   |
| `skl2onnx`          | 1.15.0         | scikit-learn モデル → ONNX 変換      |
| `onnx`              | 1.14.1         | ONNX モデルの構造定義と保存           |
| `onnxruntime`       | 最新版         | ONNX モデルの高速推論エンジン        |

---

## 🛠️ セットアップ手順

### 1. 仮想環境の作成と有効化

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows の場合: .venv\Scripts\activate


.
├── .venv/                    # 仮想環境（.gitignoreで除外推奨）
├── classifier.onnx           # 変換済みONNXモデル
├── train_and_export.py       # 学習 & ONNX変換用スクリプト
├── inference.py              # 推論スクリプト
├── requirements.txt          # 依存ライブラリ一覧（任意）
├── README.md                 # このドキュメント

