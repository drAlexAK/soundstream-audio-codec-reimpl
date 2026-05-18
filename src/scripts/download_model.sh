#!/usr/bin/env bash

huggingface-cli download "alex-kudryashov/soundstream-reimpl" \
  configs/model.yaml \
  checkpoints/best.pth \
  --local-dir pretrained
