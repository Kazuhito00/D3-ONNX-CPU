[[Japanese](README.md)/[English](README_EN.md)]

# D3-ONNX-CPU

vLLM Semantic Router Team の Decision 3.0 モデルのうち [`vllm-sr/d3-edge`](https://huggingface.co/vllm-sr/d3-edge)（0.59B）、[`vllm-sr/d3-lite`](https://huggingface.co/vllm-sr/d3-lite)（0.85B）、[`vllm-sr/d3-nano`](https://huggingface.co/vllm-sr/d3-nano)（2.21B）を、onnxruntime・tokenizers・numpy だけで動かす最小依存の推論実装です（文章と画像に対する型付き判定）。GPU でも動かせます。<br>
ベンダーの runtime（`d3_runtime.py`）が必要とする torch と transformers は使いません。

**d3-mini（5B）は未対応です**: fp32 重み（約 18GB、4.54B パラメータ）の export に空きメモリ約 25GB が必要で、今回は見送りました（マシンは 31.7GB で他の作業と共有）。詳しくは「制限事項」を参照してください。

# 特長
- 最小依存: 実行時は onnxruntime・tokenizers・numpy だけ（画像を使うときは pillow も）
- 使うスクリプトは 2 本: `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）。画像は `demo_inference_image.py`
- 3 サイズを 1 つのリポジトリ・1 つのパッケージ `d3_decide` にまとめました（`--model edge|lite|nano`）
- 判定は参照実装（ベンダー runtime を PyTorch で、backbone を fp32 に変換して実行）と一致: d3-edge 110/110 問（確率の最大差 1.3e-5）、d3-lite 110/110（4.9e-6）、d3-nano 64/64（1.6e-5）
- 設問の型は `choice`、`noul`（yes/no）、`score`（段階）。全選択肢の確率を返し、文章は生成しません
- 画像に対応（1 リクエストに何枚でも。1 枚ずつ encode して連結）。読み方はベンダーと同じ（縦横比を保ち、面積 65,536〜1,638,400 画素、32x32 画素で 1 トークン）
- 前処理（chat template、3 次元 M-RoPE の位置、画像のパッチ化、vision の 2 つの位置テーブル）を numpy と Pillow に移植し、transformers との差分ゲートを付けました
- CPU のほか、`--device cuda` で GPU（CUDA）でも動きます。実際に使われた provider を表示し、CPU への黙ったフォールバックは警告にします（`--require-gpu` ならエラー）

# このリポジトリの目的
- d3 の判定を、3〜4 個の実行時依存だけでどこまで動かせるか
- ベンダーの前処理（プロンプト、tokenizer、画像プロセッサ、rope 位置）を移植して同じ入力を作れるか
- CPU（と小さな GPU）での速度とメモリ


# 必要なもの
```
Python 3.10 以上

onnxruntime  1.20 以上   （GPU で動かすときは onnxruntime-gpu。両方を同居させない: 下記）
tokenizers   0.20 以上
numpy        1.26 以上
pillow       10.0 以上   （画像を使うときだけ）
```
実行に必要なのはこれだけです（新しい `uv sync` 環境で 131MB。`import torch` は失敗します。それが目的です）。モデルの取得は標準ライブラリだけで動きます。<br>
環境は [uv](https://docs.astral.sh/uv/) で作ります。任意のグループと extra:
- `cpu`（既定のグループ）: onnxruntime（CPU 版）。`uv sync` で入ります
- `gpu`（extra）: onnxruntime-gpu。`cpu` と同時には入れられません（「GPU で動かす」参照）
- `export`: torch、transformers 5.17.0、onnx など。自分で ONNX を作る、またはオラクルのゲートを動かすときだけ使います（実行時には使いません）

# インストール

```bash
git clone https://github.com/Kazuhito00/D3-ONNX-CPU
cd D3-ONNX-CPU

# 実行時依存だけ（onnxruntime、tokenizers、numpy、pillow）
uv sync
```

# モデルの取得
ベンダーは ONNX を公開していません。変換した ONNX は [Releases](https://github.com/Kazuhito00/D3-ONNX-CPU/releases) に置く予定です（リポジトリには含みません。Release はまだアップロードしていません）。<br>
次のスクリプトが `models/<model>/` に取得します。標準ライブラリだけで動き、git も git-lfs も不要です。
```bash
uv run download_model.py                  # 全モデル（edge、lite、nano。約 16GB）
uv run download_model.py --model edge    # 1 モデルだけ（約 2.6GB）
uv run download_model.py --model lite    # 約 4.2GB
uv run download_model.py --model nano    # 約 9.3GB
```
- 引数なしでは Release の全モデル（全グラフ、vision encoder、tokenizer、設定）を取得します。1 つだけ欲しいときは `--model` を使います
- `--seq-len 384` でそのバケットのグラフだけを取得します（重み・vision encoder・tokenizer は常に付いてきます）。文章だけなら `--exclude "vision_encoder*"` で vision encoder を省けます
- モデルごとに `manifest-<model>.json` とアセット名 `d3-<model>__...` があります。全ファイルを manifest の SHA-256 と照合し、不一致なら削除して止まります
- 1.9GiB を超えるファイルは GitHub の 2GiB 制限のため分割してあり（lite の重みは 2 分割、nano は 4 分割）、取得後に結合します。既にあるファイルは再取得せず、中断した取得は続きから再開します

自分で export する場合（Hugging Face から元のチェックポイントを固定リビジョンで取得し、全ファイルをチェックポイント自身の `MODEL_MANIFEST.json` と照合し、ベンダー runtime で golden データを作り、ONNX を出力します。torch が必要）:
```bash
UV_PROJECT_ENVIRONMENT=.venv-export uv sync --group export      # torch 環境を .venv と分ける
.venv-export/Scripts/python export/prepare_model.py --model edge --hf-dir <リポジトリ外のチェックポイント置き場>
.venv-export/Scripts/python export/prepare_model.py --model lite --seq-len 384 --seq-len 512      # 一部のバケットだけ
```
nano の export は fp32 の重み 7.5GB をメモリに載せます（export 中はその数倍。31.7GB のマシンで動きました）。ベンダー runtime での golden 作成は遅いので、`--golden-args "--stride 2"`（文章リクエストを 1 つおきに使う）を使えます。同梱の nano の golden データはこの設定で作りました。

# 使い方

### 実行例（Python）
```bash
uv run demo_inference_text.py                     # d3-edge
uv run demo_inference_text.py --model lite
uv run demo_inference_image.py --model nano
```
画像のデモは既定で `artifacts/images/sample_chart.png`（棒グラフ。「どの棒が一番高いか」「棒グラフか」を尋ねます）を使います。サンプルはスクリプトで描いた合成画像で、写真ではありません。自分の画像は `--image` で指定します。
```bash
uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
uv run demo_inference_image.py --max-pixels 50176      # 画像の面積に上限（トークンが減り速くなるが、モデルが評価された条件とは変わります）
```

```python
from d3_decide import D3Engine

engine = D3Engine("models/edge")            # リクエストが収まる最小のバケットを選びます

engine.system_one(
    state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    questions={
        "route": {"type": "choice", "instructions": "Which team should handle this request?",
                  "criteria": {"returns": "Refunds, replacements and damaged deliveries",
                               "billing": "Payments, invoices and charges",
                               "technical": "Product setup and faults"}},
        "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
        "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["Routine", "Soon", "Today"]},
    },
    images=None,                            # パス、bytes、base64 の data URL、PIL 画像のリスト
)
# {"model": "d3-edge", "answers": {"route": {"type": "choice", "choice": "returns", "probabilities": {...}, "confidence": ...}, ...},
#  "usage": {"input_tokens": ..., "output_tokens": 0}}

engine.ask(state, images, "choice", "Which team?", [("billing", "charges"), ("technical", "bugs")])   # -> 最良の選択肢の添字
```
`system_one` のリクエストとレスポンスの形は、ベンダーの `model.system_one(state=..., questions=..., images=...)` と同じです（動画は非対応）。答えられない設問は `{"type", "error", "message"}`（`invalid_question` または `max_length_exceeded`）になり、他の設問は答えます。各サイズのモデルフォルダは `models/<model>` です（変更: `--model-dir`、または環境変数 `D3_EDGE_MODEL_DIR` / `D3_LITE_MODEL_DIR` / `D3_NANO_MODEL_DIR`）。

### 設問の型
| `type` | 答え |
|---|---|
| `choice` | 選択肢のどれか（`choice`、`confidence`、`probabilities`）。`criteria` は選択肢キーから説明（または `None`）への対応 |
| `score` | 順序尺度での期待値（`score`、`legend`）。段階は 2〜10 で、最初が最低 |
| `noul` | 文が成り立つか（`noul` = yes の確率）。`criteria` で `"false"` と `"true"` に説明を付けられます |

### GPU で動かす
`--device cuda` で CUDA（CUDAExecutionProvider を先、CPU を後）、`--device auto` でビルドにある最初の GPU provider（CUDA、なければ DirectML）、既定は `cpu` です。実際に動いた provider が表示されます（Python では `engine.providers_used`）。GPU が使えないときは警告を出して CPU で動きます。**`--require-gpu` を付けるとエラーになります**。GPU の測定をするときはこれを使ってください。
```bash
uv sync --extra gpu --no-group cpu       # onnxruntime の代わりに onnxruntime-gpu（両方は入れられません）
uv run --no-sync demo_inference_text.py --device cuda --require-gpu
```
```python
engine = D3Engine("models/edge", device="cuda", require_gpu=True)
print(engine.providers_used)   # {"decoder": "CUDAExecutionProvider", "vision": "CUDAExecutionProvider"}
```
- 要件（ここで使った組み合わせ）: onnxruntime-gpu 1.31、CUDA 13.x、cuDNN 9（CUDA 13.4、cuDNN 9.20）。`onnxruntime-gpu` は CUDA の DLL を同梱しません。起動前に CUDA の `bin\x64` と cuDNN の `bin\13.2\x64` を `PATH` に入れてください（Windows の例: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64`、`C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64`）。入っていないと onnxruntime は CUDA provider を外します。警告と `--require-gpu` はそのためにあります
- `onnxruntime` と `onnxruntime-gpu` は同じディレクトリに書き込むため、同じ環境に両方を入れてはいけません。`pyproject.toml` では `cpu` グループと `gpu` extra を排他にしてあります。切り替えたあとは `uv run --no-sync` を使ってください（uv が CPU 版を戻さないように）。別の環境にしても構いません（`uv venv .venv-gpu && uv pip install --python .venv-gpu onnxruntime-gpu tokenizers numpy pillow`）
- VRAM: fp32 の重みだけで edge 2.0GB、lite 3.0GB、nano 7.5GB（vision encoder は別）。バケットごとにグラフが GPU に載ります。以下は実測したことだけを書きます
- RTX 3050 Ti Laptop（4GB）、`--require-gpu`、CUDA provider を確認、CPU 8 スレッド、マシンは静か（較正 168〜184 ms）で測定: **d3-edge は CPU の 4〜6 倍速い**（1 設問あたり L=128 で 75 ms（CPU 301 ms）、L=384 で 144 ms（CPU 880 ms）。224px 画像 1 枚 + 3 設問（L=256）で 349 ms（CPU 2,069 ms）、vision encoder は 21 ms（CPU 180 ms））。**d3-lite は速くなりません**（L=128 で 1 設問 834 ms、CPU は 769 ms。重み 3GB が 4GB のカードをほぼ埋め、一部のノードが CPU に残ります）。**d3-nano の GPU 実行は未検証**です（重み 7.5GB は 4GB に載らないため試していません）。GPU の答えは fp32 参照と一致しました（d3-edge、27 問: 答えの一致 27/27、確率の最大差 7.7e-4。演算順序が違うため CPU の結果（1.3e-5）より差が大きくなります）。より大きな GPU は測っていません
- CUDA provider では `ScatterND ... only guarantees to be correct if indices are not duplicated` という警告が出ます。このグラフの index は重複しません

# 仕組みと、系列長が固定である理由
**1 設問が 1 回の forward です。** プロンプトは system ターン（"You are a decision engine ..."）、user ターン（画像が先、続いて `State:`、`Question:`、1 トークンのコード `A`、`B`、`C`、...、`AA`、... の下に並べた選択肢）、そして空の think ブロックを持つ開いた assistant ターンです。decoder は最後のプロンプトトークンの hidden state を返し、255 x hidden の FP32 readout が 255 個のコード logits にし、答えは先頭 N 個（N = 選択肢の数）への softmax です。設問が複数あるリクエストは順番に走らせます。画像は 1 回だけ encode します。

ベンダーは softmax attention 層にプロンプト全体を見せ（`noncausal_full_attention`: 双方向、padding はマスク）、Gated DeltaNet 層は causal のままにします。export したグラフは `attention_mask` から同じマスクを作ります。

Gated DeltaNet 層は 64 トークンのチャンクを Python のループで回し、exporter はそれを**固定回数に展開**します。そのため系列長が動的なグラフは作れません。グラフは固定長 L で export し、プロンプトを L まで**右パディング**します。padding されたキーは softmax 層でマスクされ、再帰層は causal なので、読み出し位置（最後の実トークン）は padding を見ません（`verify/check_buckets.py`: 同じ設問を別のバケットで走らせた差は 2e-6）。**コストは実トークン数ではなく L に比例**するので、6 つのバケットを export し、runtime はリクエストが収まる最小のものを読み込みます:

| L | 収まるもの | 用途 |
|---:|---|---|
| 128 | 短い文章の設問（約 90〜125 トークン） | 最小。最速 |
| 256 | state が長い文章の設問、または 224px 画像 1 枚（+64 トークン）と設問 1 つ | |
| 384 | `--seq-len` の既定の開始位置 | |
| 512 | 640 x 480 の画像（300 トークン）と設問 | |
| 1024 | 長い state | |
| 2048 | 約 1.6MP までの画像（1,600 トークン）と設問 | |

- `D3Engine(model_dir, seq_len=N)` は N 以上の最小バケットから始め（既定 384）、収まらないリクエストが来るとそれより大きい最小のバケットに切り替えて、そこに留まります（古いグラフを先に解放するので、decoder が 2 つメモリに載ることはありません）。大きなバケットの分の時間を払わないよう、想定するバケットで始めてください（短い設問なら `--seq-len 128`）。最大のバケットを超えるプロンプトは `max_length_exceeded` と答えます（切り詰めません）
- バケットは**重みのファイルを 1 つ共有**します（`models/<model>/onnx/weights.onnx.data`。バイト列が同じテンソルは同じ場所から読むので、数値は変わりません）。バケットを足すと増えるのはグラフだけです（edge は 58〜64MB、lite と nano は約 165〜185MB）
- 224px の画像は 256 x 256（ベンダーの最小面積）に拡大されて 64 トークン、640 x 480 の画像は 300 トークンです

# 検証
```bash
uv run verify/run_all.py --model edge                       # 環境で動くゲートを全部（lite、nano も同様）
.venv-export/Scripts/python verify/run_all.py --model edge --hf-dir <チェックポイントのフォルダ>   # オラクルのゲートも（torch + transformers 5.17.0）
```

| ゲート | 確かめること | 必要なもの |
|---|---|---|
| check_prompt | 全 golden 設問でプロンプト文字列とトークン ID がベンダー runtime と一致 | 実行時依存 |
| verify | fp32 と bf16 の参照に対する確率と答え（文章、日本語、JSON の state、選択肢 2〜30、画像 1 枚と 2 枚） | 実行時依存 |
| check_buckets | 同じ設問を別のバケットで走らせた差、大きいバケットへの切り替え、`max_length_exceeded` | 実行時依存 |
| check_image | 画像の前処理がプロセッサ自身の `pixel_values` と一致（golden の画像） | + pillow |
| check_image_fuzz | ランダムな画像 10 サイズ（最大 3000 x 2000）で同じ確認 | + transformers |
| check_tokenizer_fuzz | ランダムな多言語文 20,000 件とプロンプトで、`tokenizers` と transformers の tokenizer を比較 | + transformers |
| check_tables_torch | numpy 版の vision 位置テーブルと M-RoPE を transformers と比較 | + torch |
| tests/ | 設問、プロンプト、答え、画像の幾何、provider 選択の単体テスト（`python -m unittest discover -s tests`） | 実行時依存 |

`artifacts/<model>/golden.json` は**ベンダー runtime**（`export/make_golden.py`、CPU、transformers 5.17.0）を実際に走らせて作った正解データで、配布どおり（bf16 の backbone）と、backbone を fp32 にした場合の 2 通りを持ちます。ONNX のグラフは fp32 なので、fp32 の実行が厳密なオラクル、bf16 の実行は bf16 の丸めだけで答えがどれだけ動くかを示します。リクエスト（`export/cases.py`）は英語と日本語の state、JSON の state、選択肢 2〜30、choice / noul / score、画像 1 枚と 2 枚を含みます。

| モデル | 設問数 | fp32 参照との比較: 答えの一致、確率の最大差 | bf16（配布どおりのベンダー）との比較: 答えの一致、確率の最大差 |
|---|---:|---|---|
| d3-edge | 110（39 リクエスト） | 110/110、1.25e-5 | 108/110、3.3e-2 |
| d3-lite | 110（39 リクエスト） | 110/110、4.9e-6 | 110/110、3.7e-2 |
| d3-nano | 64（23 リクエスト: 文章は 1 つおき、画像 7 リクエストは全部） | 64/64、1.6e-5 | 64/64、3.6e-2 |

d3-edge の bf16 との不一致 2 問は、bf16 の丸め自体が答えを動かした例です（3 モデルとも bf16 の丸めで確率が最大 0.03〜0.04 動きます）。画像の前処理は、プロセッサの `pixel_values` との最大差が、ランダムな画像 10 サイズ（`check_image_fuzz`）で 2 階調（平均 1e-5）、サンプル画像では厳密に 0 です。vision の位置テーブルと M-RoPE は transformers と完全に一致し（40 グリッド、60 のプロンプト配置）、tokenizer はランダム 30,000 件で一致しました。

# 性能
CPU: Intel Core i7-12800H、8 スレッド、onnxruntime 1.31、fp32。ベンチの GLiNER 較正（`runner.calibrate`。静かなマシンで約 199 ms）は各モデルの前後で 168〜184 ms で、マシンは静かでした。ウォームアップ後の 5 回の中央値で、tokenize を含む 1 設問あたりの時間です。「読み込み」はグラフのセッション作成にかかる時間です。

| モデル | L=128 | L=384 | L=512 | セッション読み込み | ピーク RAM |
|---|---:|---:|---:|---:|---:|
| d3-edge（0.59B） | 301 ms | 880 ms | 1,187 ms | 8〜9 秒 | 1.8〜2.8GB |
| d3-lite（0.85B） | 769 ms | 2,216 ms | 3,003 ms | 31〜34 秒 | 3.3〜5.1GB |
| d3-nano（2.21B） | 1,263 ms | 3,786 ms | 5,059 ms | 41〜64 秒 | 7.5〜9.3GB |

（1 設問あたり。108〜128 トークンの文章の設問、1 リクエスト 3 設問、列のグラフで実行。）画像つき: 224px の画像 1 枚（64 トークン）と 3 設問（L=256）で、edge は 1 設問 690 ms + vision encoder 180 ms（1 回）、lite は 1,578 ms + 162 ms、nano は 2,753 ms + 516 ms。640 x 480 の画像（300 トークン）と 3 設問（L=512）では、edge は 1 設問 1,629 ms + 928 ms、lite は 3,482 ms + 904 ms、nano は 6,136 ms + 2,830 ms（ピーク RAM は edge 3.2GB、lite 5.5GB、nano 10.0GB）。コストはバケットに従うので、設問が収まる最小のバケットで始めてください。GPU の数値は「GPU で動かす」を参照してください。

# 制限事項
- **d3-mini（5B）は未対応です。** export には空きメモリ約 25GB が必要で（fp32 の重み約 18GB と exporter のコピー）、ベンダー runtime での fp32 の golden 作成にもさらに約 18GB かかります。他の作業と共有する 31.7GB のマシンでは見送りました。重みと export のコードは他の 3 つと同じ経路で（`export/models.py` に項目を足すだけのはず）、ただし試していないため、動くかどうかについては何も主張しません
- 動画は非対応です（ベンダーのモデルは読めますが、この移植は対応しません）。画像はパス、bytes、base64 の data URL、PIL 画像から読みます。URL は取得しません
- ベンダーの任意機能 `permutation_average` は実装していません（ベンダー runtime でも既定は off です）
- 系列長はグラフごとに固定です（上記）。最大のバケット（2048）を超えると設問は `max_length_exceeded` になります
- fp32 の重みだけです。ベンダーは bf16 の backbone で動かすので、同点に近い設問では答えが変わり得ます（上の表）。d3-nano の重みは fp32 で 7.5GB です
- `tokenizer.json` の pre-tokenizer は `[\p{L}\p{M}]+` ですが、transformers 5.17.0 の `Qwen2Tokenizer`（ベンダー runtime が使うもの）はその正規表現を `\p{L}+` に置き換えます。この移植はベンダーの挙動に合わせてあり（`load_tokenizer`）、違いが出るのは結合文字を含む文章だけです
- `optimize=True` で export してはいけません（展開されたグラフでは optimizer が非常に遅くなります）
- ベンダーの model card にある速度と精度の数値は、ここでは再現していません
- d3-nano の fp32 golden データは、他の 2 つの 110 問のうち 64 問（文章は 1 つおき）です。d3-mini にはありません

# プロジェクト構成

```text
README.md / README_EN.md   # README（日本語 / 英語）
LICENSE                    # Apache-2.0
LICENSE-VLLM-SR            # ベンダーの Apache-2.0 全文（無変更）
NOTICE                     # 帰属表示: この変換の説明 + ベンダーの NOTICE をそのまま
pyproject.toml             # 依存（実行時: tokenizers、numpy、pillow + group cpu。extra gpu。group export）
download_model.py          # モデル取得（GitHub Releases、SHA-256 照合、分割ファイルの結合）
demo_inference_text.py     # 文章の推論デモ
demo_inference_image.py    # 画像の推論デモ
d3_decide/                 # 推論の実装（export/ を import しません）
  runtime.py               #   ORT セッション + バケット + 右パディング + readout + provider + system_one / ask
  format.py                #   設問、プロンプト、chat template、答え（d3_format.py と設問処理の移植）
  chat.py                  #   3 次元 M-RoPE の位置
  image.py                 #   Pillow の読み込み + smart_resize + パッチ化 + vision の 2 つの位置テーブル
  paths.py                 #   モデルフォルダ
export/                    # 重い依存（torch / transformers / onnx）
  prepare_model.py         #   チェックポイント取得 + golden + export + 重みの共有
  export_onnx.py           #   decoder グラフ（L ごと）+ vision encoder
  graph_modules.py         #   export する torch モジュール（fp32 のオラクルでもあります）
  share_external_data.py   #   バケットが重みを 1 つ共有（同一テンソルのみ）
  make_golden.py, cases.py #   ベンダー runtime から作る正解データ
  models.py                #   チェックポイントと固定リビジョン
release/make_release.py    # models/ から Release 用ファイルを作る（分割、SHA-256、manifest-<model>.json）
verify/                    # 検証ゲート（run_all.py でまとめて実行）と bench.py
tests/                     # 単体テスト
artifacts/<model>/         # golden.json（正解データ）、verify_run.log。artifacts/images はサンプル画像、artifacts/bench_smoke はベンチのスモークのログ
models/<model>/            # ONNX、tokenizer、設定（gitignore）
```

# 参考
- [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge)（リビジョン `18d61df7`）、[vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite)（`a8d25a91`）、[vllm-sr/d3-nano](https://huggingface.co/vllm-sr/d3-nano)（`3aeb1c22`）: 元のモデル（vLLM Semantic Router Team）
- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)（d3-edge と d3-lite のベース。d3-edge は 24 層のうち 12 層を持ちます）、[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)（d3-nano のベース）: Alibaba Cloud
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)

# ライセンス
D3-ONNX-CPU は [Apache-2.0 license](LICENSE) です。<br>
モデルの重みは元のモデルのライセンスに従います（[vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge)、[d3-lite](https://huggingface.co/vllm-sr/d3-lite)、[d3-nano](https://huggingface.co/vllm-sr/d3-nano) は、各 model card、`LICENSE`、Hugging Face のタグが示すとおり Apache-2.0 です。ベースの Qwen/Qwen3.5-0.8B と Qwen/Qwen3.5-2B も Apache-2.0 です）。ベンダーの `LICENSE`（[LICENSE-VLLM-SR](LICENSE-VLLM-SR)）と `NOTICE` の文面は変更せずに保持しています（[NOTICE](NOTICE) はこの変換の説明を加え、ベンダーの文面を再掲しています）。Release には、各モデルの ONNX ファイルと並べて両方を入れます。

## 変換したファイルについて
Release の ONNX ファイルは、[vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge)、[d3-lite](https://huggingface.co/vllm-sr/d3-lite)、[d3-nano](https://huggingface.co/vllm-sr/d3-nano)（vLLM Semantic Router Team。Apache-2.0。上記のリビジョン）のチェックポイントを、このリポジトリの `export/export_onnx.py` で変換したものです（decoder と vision encoder は別のグラフとして export し、decoder は系列長ごとに 1 つ、重みは長さの間で共有します。`readout.safetensors` の readout は decoder グラフに組み込み、`noncausal_full_attention` のマスクはグラフ内で作り、vision の位置テーブルはグラフの外で計算します）。重みは fp32 で出力しています（元は bf16 で、bf16 は fp32 の厳密な部分集合なので重みそのものは変わりません）。ベンダーは bf16 で計算するため、数値はわずかに変わります（「検証」参照）。元のモデルの提供者は、この変換を推奨せず、動作も保証していません。

model card には学習データの記述がなく、そこにある評価値はベンダー自身のものです。ここでは再現していません。モデルに頼る前に、元の model card を読んでください。
