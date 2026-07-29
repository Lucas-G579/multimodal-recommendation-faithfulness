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

