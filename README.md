# SoundStream Audio Codec Reimplementation

This repository contains a reimplementation of the SoundStream neural audio codec for speech.

For 16kHz audio files without denoising.

The model is trained on LibriSpeech `train-clean-100` and evaluated on LibriSpeech `test-clean`.

This implementation includes from original paper:

- SoundStream encoder, RVQ quantizer, and decoder;
- waveform and STFT discriminators;
- STOI and NISQA evaluation;
- inference and demo notebook support.

with only difference in architecture being generator embeddings sizes and random first-batch initialization instead of k-means:
```
embedding_dim: 256
```

## Installation

```bash
git clone https://github.com/drAlexAK/soundstream-audio-codec-reimpl.git
cd soundstream-audio-codec-reimpl
pip install -r requirements.txt
```

## Demo

To try out the model yourself or to get more familiar with it, use:

```text
demo.ipynb
```

## Model

The final checkpoint and resolved model config are stored on Hugging Face.

```bash
bash src/scripts/download_model.sh
```

It will create a repository with all information needed for inference.

```text
pretrained/
  configs/model.yaml
  checkpoints/best.pth
```

## Training

Model was trained in a 3 staged way contrary to the original paper. You can recreate it using this configs:

```bash
python3 train.py -cn training/stage_1
python3 train.py -cn training/stage_2
python3 train.py -cn training/stage_3
```

the only difference between them is this loss coeffitient history:

### Loss coefficient schedule

  `lambda_adv`: `1.0 -> 0.1 -> 0.01`

  `lambda_feat`: `100.0 -> 10.0 -> 1.0`

  `lambda_rec`: `1.0 -> 1.0 -> 1.0`

  `lambda_commit`: `1.0 -> 1.0 -> 1.0`

  Stage boundaries:

```text
stage 1: epochs 1-12
stage 2: epochs 13-30
stage 3: epochs 31-35
```

best checkpoint beign 34th.

## Inference And Evaluation

You can evaluate the model on LibriSpeech `test-clean`:

```bash
python3 evaluate.py
```

Or run checkpoints on different datasets using:

```bash
python3 inference.py datasets=... dataloader=...
```

## Citation
This project uses the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) by Petr Grinberg as the base project structure.
