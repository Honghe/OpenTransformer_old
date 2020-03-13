# OpenTransformer

This is a speech transformer model for end-to-end speech recognition.

## Requirements
Pytorch >= 1.2.0

Torchaudio >= 0.3.0

## Function

- Speech Transformer

- Label Smoothing

- Share weights of Embedding with output softmax layer

- Data Augmentation([SpecAugument](https://arxiv.org/abs/1904.08779))

- Extract Fbank features in a online funshion

- Visualization based Tensorboard

- Batch Beam Search with Length Penalty

- Multiple Optimizers and Schedulers

- Multiple Activation Functions in FFN

- Multi GPU ([dp](https://pytorch.org/docs/stable/nn.html#dataparallel), [ddp](https://pytorch.org/docs/stable/nn.html#distributeddataparallel))

- Mixed Precision Training based [apex](https://github.com/NVIDIA/apex)

## To Do
- LM Shallow Fusion

## Prepare
vocab  
只使用id即可，词表不需要加<blank>和<unk>等占位符。
与配置文件`transformer.yaml`的关系是：vocab_size＝词表+3。

character idx
```
我 3
你 4
```

character
```
BAC009S0764W0139 国 家 统 计 局 的 数 据 显 示
BAC009S0764W0140 其 中 广 州 深 圳 甚 至 出 现 了 多 个 日 光 盘
BAC009S0764W0141 零 三 年 到 去 年
BAC009S0764W0142 市 场 基 数 已 不 可 同 日 而 语
BAC009S0764W0143 在 市 场 整 体 从 高 速 增 长 进 入 中 高 速 增 长 区 间 的 同 时
BAC009S0764W0144 一 线 城 市 在 价 格 较 高 的 基 础 上 整 体 回 升 并 领 涨 全 国
BAC009S0764W0145 绝 大 部 分 三 线 城 市 房 价 仍 然 下 降
BAC009S0764W0146 一 线 楼 市 成 交 量 激 增
BAC009S0764W0147 三 四 线 城 市 依 然 冷 清
```
if you want to compute features online, please make sure you have a wav.scp file.
```
# wav.scp
# id path
BAC009S0764W0139 /data/aishell/wav/BAC009S0764W0139.wav
```

## Preprocess Aishell
```
python preprocess_scp_and_character.py
```

## Train
Loss降到0.2及以下。

- Single GPU

```python
python run.py -c egs/aishell/conf/transformer.yaml
```
- Multi GPU Training based DataParallel
```python
python run.py -c egs/aishell/transformer.yaml -p dp -n 2
```
- Multi GPU Training based distributeddataparallel
```python
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 run.py -c egs/aishell/transformer.yaml -p ddp -n 2
```

## Eval
```python
python eval.py -m model.pt
```

## Experiments
Our Model can achieve a CER of 7.1% without CMVN, any external LM and joint-CTC training on [AISHELL-1](http://www.openslr.org/33/), which is better than 7.5% of Chain Model in Kaldi.

## Acknowledge
OpenTransformer refer to [ESPNET](https://github.com/espnet/espnet).
