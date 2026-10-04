# Pinned DSFormer carrier launch

The registered checkpoint was absent and no official downloadable DSFormer
checkpoint was found in the Fair PSG repository/project documentation. The
carrier was therefore started from the pinned `masks-loc-sem.json` recipe.

Environment: `/home/user5/anaconda3/envs/adframe/bin/python` with torch
2.3.1+cu121, torchvision 0.18.1+cu121, transformers 4.57.6, h5py, and
tensorboard. The missing packages were installed into this existing
environment.

Command:

```bash
bash experiments/counterfactual_relation_verification_v1/launch_pinned_carrier_current_paths.sh
```

Runtime paths use the verified `/data1/liuhaoran/psg` dataset and checkpoint
root, GPU 7, 40 epochs, batch size 32, 4 workers, and the pinned source
revision `2497be40bd8bd0ca6c2ad25be1522ebf42e6623b`.

At record time the process was alive, had written the config/args/commit and
TensorBoard event files, and had reached 1,081 training steps without error.
