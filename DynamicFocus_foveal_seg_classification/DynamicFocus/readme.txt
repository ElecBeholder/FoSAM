1. install requirement
terminal:
pip install -r req.txt

2. goto preset.py
set all the required directory, make sure you have enough space
data_c_cache > 6GB

3.1 execute following to sampole data
goto b_preprocess_cityscapes.py
console:
ppcc = PreprocessCityscape(dataset_partition='train', downsample_degree=3, use_cache=True)
ppcc.prep_N_samples(samples_per_class=1)

3.2 load training data
namekeys_all = []
X_rgb_Bx3xHxW, X_focus_Bx2, Y_BxHxW = ppcc_train_2.load_all_samples(namekeys_all=namekeys_all)

3.3 load to see the croped rgba image for visualization
x_4xHxW = ppcc_train_2.load_rgba_sample_by_namekey(namekeys_all[32])

4. run training script
goto nn_C_manager.py
console:
mm.train()

5. run tensor board
terminal:
tensorboard --logdir=a_training_records
