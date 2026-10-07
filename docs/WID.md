# WID baseline

[Back to baselines](../baselines/README.md) | [PIU appendix](https://arxiv.org/html/2605.22311v2)

WID trains on real forget images. This Arc2Face adaptation uses a frozen teacher and an IR-SE50 recognizer for its identity loss.

```bash
python baselines/run_all.py --identity-id 512 --methods wid \
  --data-dir data/celebahq_512 --use-anchor-overrides \
  --output-dir outputs/wid_512
```

The launcher prepares missing paper data and images. IR-SE50 [weights](https://huggingface.co/AIRI-Institute/StyleFeatureEditor/blob/5a50cb1b78da946be8163b8b255d486996fd2c73/pretrained_models/model_ir_se50.pth) download on first use. For local PyTorch weights, add `--wid-args='--identity-checkpoint /path/to/model_ir_se50.pth'`.

For recomputed data, pass `--data-dir data/celebahq_512_recomputed` and omit `--use-anchor-overrides`. ONNX ArcFace weights cannot replace the IR-SE50 checkpoint.

Defaults: 100 updates, learning rate `5e-6`, identity weight `0.1`, zero weight decay, and effective batch 64 (GPU batch 4, accumulation 16). Each forget image has its own identity target; this corrects the old anchor-mean target. The identity loss uses single-step reconstruction, not iterative denoising.

Results in `outputs/wid_512` include the checkpoint, metrics, curves, and before/after images. The original paper's scores are not guaranteed by the shared demo split and defaults alone.
