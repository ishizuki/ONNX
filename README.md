
python3 -m venv .venv
source .venv/bin/activate

# パッケージのインストール
pip install --upgrade pip
pip install scikit-learn==1.3.2 skl2onnx==1.15.0 onnx==1.14.1 onnxruntime
