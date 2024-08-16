# nsfw_model_onnx_sample
NSFW判定モデルの[GantMan/nsfw_model](https://github.com/GantMan/nsfw_model)のPythonでのONNX推論サンプルです。<br>
変換自体を試したい方は、Google Colaboratory上で[Convert2ONNX.ipynb](Convert2ONNX.ipynb)を使用ください。<br>

![image](https://github.com/user-attachments/assets/c7ab3ea9-15c9-4f12-a316-90d3d975dfb8)

# Requirement 
* OpenCV 4.5.3.56 or later
* onnxruntime 1.11.0 or later

# Convert
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/nsfw_model_onnx_sample/blob/main/Convert2ONNX.ipynb)<br>
Colaboratoryでノートブックを開き、上から順に実行してください。

# Demo
デモの実行方法は以下です。
```bash
python demo_onnx.py
```
* --image<br>
画像パスの指定<br>
デフォルト：sample.jpg
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/nsfw_mobilenet2_224x224.onnx

# Reference
* [GantMan/nsfw_model](https://github.com/GantMan/nsfw_model)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
nsfw_model_onnx_sample is under [MIT License](LICENSE).

# Note
サンプルの画像は[ぱくたそ](https://www.pakutaso.com/)様の「[こちらを見上げる子猫](https://www.pakutaso.com/20240119016post-50296.html)」を使用しています。
