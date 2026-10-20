# 手話単語認識デモ（Web版）

スマホやPCのブラウザで、カメラ映像から WLASL100 の手話単語を認識するデモです。
すべてブラウザの中で動き、映像はどこにも送信されません。

## ファイル

| ファイル | 中身 |
|---|---|
| index.html | 画面 |
| app.js | カメラ・MediaPipe Holistic・判定の処理 |
| features.js | 393次元の特徴量の計算（学習時のPythonと同じ計算。Pythonとの一致を確認済み） |
| dgtsta_wlasl100.onnx | 認識モデル（サーバーの `deploy/` からコピーして置く） |
| glosses.json | 単語の一覧（同上） |

## 使っているもの

- MediaPipe Tasks Vision 0.10.14（HolisticLandmarker、GPUが使えなければCPU）
- ONNX Runtime Web 1.18.0（wasm、1スレッド）
- どちらもCDN（jsDelivr）から、Holisticのモデルは Google のサーバーから読み込みます

## 動かし方

### PCで試す
このフォルダで次を実行し、Chromeで http://localhost:8000 を開きます（localhost ならカメラが使えます）。

    python -m http.server 8000

### iPhoneで試す
カメラを使うには HTTPS が必要です。このフォルダの中身（テスト動画とテストのラベルは除く）を
GitHub Pages や Netlify などの静的ホスティングに置き、Safari で開きます。

## 精度の確認
「精度の確認」を開き、test_labels.json とテスト動画（video_id.mp4）を選んで「評価を開始」を押すと、
学習時と同じ正解率が出るかを確認できます。WLASLの動画は再配布が禁止されているので、
テスト動画とテストのラベルは公開する場所に置かないでください。
