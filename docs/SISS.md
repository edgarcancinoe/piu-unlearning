# SISS baseline

[Back to baselines](../baselines/README.md) | [PIU Appendix C.1](https://arxiv.org/html/2605.22311v2#A3.SS1)

SISS trains on real forget and retain images. The demo downloads the hosted images and prepares their manifest if needed.

```bash
piu-demo --method siss --identity-id 512 --data-dir data/celebahq_512
```

For the paper comparison split, run the [comparison launcher](../baselines/README.md) with `--methods siss --use-anchor-overrides`. SISS itself does not use an anchor.

Defaults: 60 updates, learning rate `5e-6`, beta `0.1`, 1,000 retain identities, and effective batch 64. The forget-gradient scale is computed per group of 16 images. For less GPU memory, add `--batch-size 4 --gradient-accumulation-steps 16` to the command above. Use `--beta` to change forgetting strength.

Results in `outputs/siss_demo` include the checkpoint, metrics, curves, and before/after images.

**Reproduction note:** During the final reimplementation, we found that an incorrect assumption about the diffusion scheduler's timestep range caused the severe SISS failure reported in PIU. This implementation samples from the full training range and fixes that issue. SISS now forgets more effectively, but image quality still degrades. The paper reports the original runs.
