# `artifacts/` — dumped tensors, NOT model checkpoints

Both files here are `.pt` (same extension PyTorch also uses for model
weights) but are **not** trained models — they're intermediate/input
tensors saved from a single manual run of `notebooks/CP_Fusion_MODELS.ipynb`,
kept for quick reloading without re-running the notebook:

| file | what it actually is | produced by |
|---|---|---|
| `dataset0_cav0_4cams.pt` | the raw 4-camera input batch tensor for dataset scenario 0, cav 0 | notebook cell that loads a `Dataset`/`DataLoader` sample |
| `peer_feature.pt` | `torch.save(intermediate.detach().cpu(), ...)` — a saved intermediate activation tensor from pushing that input through the LSS encoder | later cell in the same notebook |

**Model checkpoints live elsewhere:**
- Official HEAL baselines (not trained here): `../../checkpoints/` (repo root).
- This project's own trained runs: `../../opencood/logs/<run_name>/net_epoch_bestval_at*.pth`.
