#!/usr/bin/env bash

hf download "alex-kudryashov/soundstream-reimpl" \
  configs/model.yaml \
  checkpoints/best.pth \
  --local-dir pretrained
