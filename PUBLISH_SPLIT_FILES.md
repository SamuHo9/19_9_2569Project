# GitHub publish note: split file over the 2 GB LFS limit

The current project snapshot contains `Model/_training_cuda_site/torch/lib/dnnl.lib`,
which is 2,325,850,314 bytes. GitHub rejected the original object because its Git LFS
per-file limit is 2 GiB. The bytes are preserved in two LFS-managed parts:

- `Model/_training_cuda_site/torch/lib/dnnl.lib.part01`
- `Model/_training_cuda_site/torch/lib/dnnl.lib.part02`

To reconstruct the original file after cloning, run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/restore_split_dnnl.ps1
```

The restore script checks SHA-256 before replacing the target. Expected SHA-256:
`b6ffe087a5ee14a206c56ddaa37380bcdcf8ec4ba7c75aec895cb895d306f05e`.
