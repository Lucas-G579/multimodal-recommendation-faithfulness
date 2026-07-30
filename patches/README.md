# Upstream patches

`external/` is intentionally excluded from Git. Reproducible changes to external
research code are stored here.

## MMRec-save-best.patch

The downloaded MMRec `Trainer.fit(..., saved=True)` accepts a save flag but never
writes a checkpoint. The patch saves the best validation state without changing
training or evaluation calculations.

Apply from the MMRec repository root:

```powershell
git apply ..\..\patches\MMRec-save-best.patch
```

## MMRec-native-scatter-add.patch

MGCN uses `torch_scatter` for one degree-aggregation operation. The patch
replaces it with the equivalent native PyTorch 2.3 `Tensor.scatter_add_`, avoiding
an unnecessary compiled dependency on Windows.

Apply from the MMRec repository root:

```powershell
git apply ..\..\patches\MMRec-native-scatter-add.patch
```
