# 2026-09-12 bootstrap record

- Created the isolated RelSupport-PSG experiment root.
- Pinned Fair PSG source revision `2497be40bd8bd0ca6c2ad25be1522ebf42e6623b`.
- Located a public mirror of the full OpenPSG annotation on Hugging Face.
- The original SharePoint dataset link redirects to a login-protected 403.
- No PSG assets or DSFormer checkpoint existed locally at discovery time.
- All eight A100 80 GB GPUs were idle, but P0 begins with CPU diagnostics and a
  single-GPU segmentation smoke only.
- `/data2` had about 163 GB free and `/data1` about 106 GB free.  The existing
  `/data2/liuhaoran/datasets` symlink was broken, so new assets use
  `/data2/liuhaoran/psg_data` without modifying that user-owned symlink.
- Downloaded and verified `psg.json` (SHA256
  `d4a8c6df10198fa2a22f56d4c2039aa2b20bdc002db5c0be8ee34a36a06817c1`).
  It contains 48,749 rows and 2,186 test images, all from COCO `val2017`.
- Created `/data2/liuhaoran/venvs/fair_psg_p0` from the working base stack and
  installed local NumPy 1.26.4 / Accelerate 1.10.1 overrides.  This fixes a
  pre-existing NumPy/Accelerate import mismatch without mutating the base env.
- Fair PSG segmentation and DSFormer inference entry points now import and show
  help successfully in that environment.
- Launched the first evidence run at 17:03 Asia/Shanghai:
  `psg-test-mask2former-swin-large-finalpanoptic-smoke8-seed0-r1` on GPU 0.
