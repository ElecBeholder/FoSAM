
# if show:
#     labelid_pred_BxHxW = torch.argmax(Y_pred_BxKxHxW, dim=1)
#     labelid_real_BxHxW = torch.argmax(Y_BxKxHxW, dim=1)
#
#     alpha_pred_BxHxW = (labelid_pred_BxHxW != 39).to(torch.float32)
#     alpha_real_BxHxW = (labelid_real_BxHxW != 39).to(torch.float32)
#
#     img_pred_Bx4xHxW = torch.concat([X_Bx6xHxW[:, :3, :, :], alpha_pred_BxHxW[:, None, :, :]], dim=1)
#     img_real_Bx4xHxW = torch.concat([X_Bx6xHxW[:, :3, :, :], alpha_real_BxHxW[:, None, :, :]], dim=1)
#
#     for bodyname, bid in zip(names, bids):
#         # plt_imgshow(img_pred_Bx4xHxW[bid], title=f"bid{bid}_pred")
#         # plt_imgshow(img_real_Bx4xHxW[bid], title=f"bid{bid}_real")
#         imgs = [img_pred_Bx4xHxW[bid], img_real_Bx4xHxW[bid]]
#         titles = [f"{bodyname}_pred", f"{bodyname}_real"]
#         plt_multi_imgshow(imgs, titles, (1, 2))
#         plt.savefig(os.path.join(preset.path_training_records, f'{bodyname}.predict.png'))
#         plt.close('all')


# args_kwargs_focal = ((mc_gpu['ys_pred_gs_BxKxHSxWS_bidxs'], mc_gpu['grid_pred_BxHSxWSx2_bidxs'], mc_gpu['gpu_y_Bx1xHxW'][bidxs].squeeze(1)), {})
# args_kwargs_edge = ((mc_gpu['xs_pred_gs_Bx3xHSxWS_bidxs'], mc_gpu['gpu_x_Bx3xHxW'][bidxs], mc_gpu['gpu_y_Bx1xHxW'][bidxs].squeeze(1), mc_gpu['dm_pred_Bx1xHSxWS_bidxs']), {})
# tensor_train_loss = loss_fctn([args_kwargs_focal, args_kwargs_edge])
avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)

y_real_gs_BxHSxWS = torch.argmax(avg_pool(F.one_hot(mc_gpu['gpu_y_Bx1xHxW'][bidxs].squeeze(1), num_classes=K).permute(0, 3, 1, 2).to(torch.float32)), dim=1)

args_kwargs_focal = ((mc_gpu['ys_pred_gs_BxKxHSxWS_bidxs'], y_real_gs_BxHSxWS), {})
args_kwargs_edge = ((mc_gpu['xs_pred_gs_Bx3xHSxWS_bidxs'], mc_gpu['gpu_x_Bx3xHxW'][bidxs], mc_gpu['gpu_y_Bx1xHxW'][bidxs].squeeze(1), mc_gpu['dm_pred_Bx1xHSxWS_bidxs']), {})

if show:
    namekeys_bidxs = [namekeys_all[bidx] for bidx in bidxs]

    images = []
    titles = []
    for i, bidx in enumerate(bidxs):
        namekey = namekeys_bidxs[i]
        x_rgba_4xHxW = ppcc.load_rgba_sample_by_namekey(namekey)
        images.extend([x_rgba_4xHxW, mgpu['xs_Bx3xHSxWS'][i], mgpu['dm_Bx1xHSxWS'][i]])

        namekey_parts = namekey.split('_')
        namekey_wrap = f"{'_'.join(namekey_parts[:-4])}\n{'_'.join(namekey_parts[-4:])}"
        titles.extend([f'{namekey_wrap}\n256x512.real', f'{namekey_wrap}\n64x128.sample', f'{namekey_wrap}\n64x128.densitymap'])

    fidxs = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    plt_multi_imgshow([images[fidx] for fidx in fidxs], [titles[fidx] for fidx in fidxs], (3, len(bidxs)))
    plt.show(block=True)

if (ep + 1) % 100 == 0:
    pass
    # label2names = pp_cc.get_all_label2names()
    # for label, names in label2names.items():
    #     name = random.choice(names)
    #     X_rgb_1x3xHxW, X_focus_1x2, Y_1xHxW = pp_cc.load_a_sample(name)
    #     Y_hat = mm.predict(X_rgb_1x3xHxW, X_focus_1x2, Y_1xHxW , names=[name], bids=[0], show=True)





if __name__ == '__main__':
    str_ts = datetime.now().strftime('D%y%m%d_T%H%M%S')

    in_channels = 3  # 例如，输入通道数为6
    out_channels = 41  # 例如，输出通道数为40
    H = ppcc.HD  # 256
    W = ppcc.WD  # 512
    downsample_factor = 4
    kernel_size = 32 + 1
    B = 10
    injectD = 2
    K = 41

    celoss_fctn = nn.CrossEntropyLoss()

    mode = 'metrix'
    methodname = 'SegerZoom_DoubleMSELossStd'

    model_name = f"D240922_T230026_SegerZoom_DoubleMSELossStd_256x512"

    show = True

    if methodname == 'SegerZoom_FocalLossStd':
        loss_fctn = FocalLossStd()

        model_select = SegerZoom(in_channels=in_channels, out_channels=out_channels, H=H, W=W, downsample_factor=downsample_factor, kernel_size=kernel_size)
        calc_model_memsize(model_select)
    elif methodname == 'SegerZoom_ObjDeformedJointLossStd':
        loss_fctn = ObjDeformedJointLossStd()

        model_select = SegerZoom(in_channels=in_channels, out_channels=out_channels, H=H, W=W, downsample_factor=downsample_factor, kernel_size=kernel_size)
        calc_model_memsize(model_select)
    elif methodname == 'SegerZoom_DoubleMSELossStd':
        loss_fctn = DoubleMSELossStd()

        model_select = SegerZoom(in_channels=in_channels, out_channels=out_channels, H=H, W=W, downsample_factor=downsample_factor, kernel_size=kernel_size)
        calc_model_memsize(model_select)

    elif methodname == 'ave':
        loss_fctn = FocalLossStd()

        model_select = Seger_Ave(in_channels=in_channels, out_channels=out_channels, H=H, W=W, downsample_factor=downsample_factor, kernel_size=kernel_size)
        calc_model_memsize(model_select)

    script_mode = 'metrics'
    if not model_name:
        model_name = f'{str_ts}_{model_select.__class__.__name__}_{loss_fctn.__class__.__name__}_{H}x{W}'
        show = False
        script_mode = 'train'

    script_mode = 'plot'
    mm = ModelManager(model_select, model_name, MM_device_gpu, refresh=False)
    if script_mode == 'train':

        mm.train(loss_fctn_seg=loss_fctn, epoch_offset=900, save_per_N_epoch=50, show=show)
    elif script_mode == 'metrix':
        mgpu = MemCache()
        ppcc_val = PreprocessCityscape(dataset_partition='val', downsample_degree=2)

        mgpu['x_Bx3xHxW'], mgpu['x_Bx2'], mgpu['y_Bx1xHxW'] = ppcc_val.load_all_samples()

        take = None
        batch_size = 5
        y_pred_BxHxW = mm.predict(mgpu['x_Bx3xHxW'][:take], mgpu['x_Bx2'][:take], batch_size=batch_size)

        print(y_pred_BxHxW.shape)
        print(mgpu['y_Bx1xHxW'][:take].shape)

        mm.get_metrics(y_pred_BxHxW, mgpu['y_Bx1xHxW'][:take], K=K, labels=ppcc.idx2label)
    elif script_mode == 'plot':
        mgpu = MemCache()

        ppcc_train = PreprocessCityscape(dataset_partition='train', downsample_degree=2)
        namekeys_all = []
        mgpu['x_Bx3xHxW'], mgpu['x_Bx2'], mgpu['y_Bx1xHxW'] = ppcc_train.load_all_samples(namekeys_all=namekeys_all)

        B, _, H, W = mgpu['x_Bx3xHxW'].shape

        random.seed(24)
        idxs_rand = random.choices(range(B + 1), k=1)

        for idx in idxs_rand:
            print(namekeys_all[idx])

        mm.plot_figure(mgpu['x_Bx3xHxW'][idxs_rand], mgpu['x_Bx2'][idxs_rand], mgpu['y_Bx1xHxW'][idxs_rand])

    # # mc_gpu['gpu_x_Bx3xHxW'], mc_gpu['gpu_x_Bx2'], mc_gpu['gpu_y_Bx1xHxW'] = ppcc.load_a_sample(namekey="train_strasbourg_000001_031427_96x422")
    # #
    # # mm.predict(mc_gpu['gpu_x_Bx3xHxW'], mc_gpu['gpu_x_Bx2'], mc_gpu['gpu_y_Bx1xHxW'])
    #
    #
    # # X, Y = ppcc.load_a_sample('bremen_000273_000019_270x832_static')
    # # Y_hat = mm.predict(X, Y, bodynames=['bremen_000273_000019_270x832_static'], bids=[0])
    # # print(Y_hat.shape)


    """
    elif script_mode == 'metrix':
    mgpu = MemCache()
    ppcc_val = PreprocessCityscape(dataset_partition='val', downsample_degree=2)

    mgpu['x_Bx3xHxW'], mgpu['x_Bx2'], mgpu['y_Bx1xHxW'] = ppcc_val.load_all_samples()

    take = None
    batch_size = 5
    y_pred_BxHxW = mm.predict(mgpu['x_Bx3xHxW'][:take], mgpu['x_Bx2'][:take], batch_size=batch_size)

    print(y_pred_BxHxW.shape)
    print(mgpu['y_Bx1xHxW'][:take].shape)

    mm.get_metrics(y_pred_BxHxW, mgpu['y_Bx1xHxW'][:take], K=K, labels=ppcc.idx2label)

elif script_mode == 'plot':
mgpu = MemCache()

ppcc_train = PreprocessCityscape(dataset_partition='train', downsample_degree=2)
namekeys_all = []
mgpu['x_Bx3xHxW'], mgpu['x_Bx2'], mgpu['y_Bx1xHxW'] = ppcc_train.load_all_samples(namekeys_all=namekeys_all)

B, _, H, W = mgpu['x_Bx3xHxW'].shape

random.seed(24)
idxs_rand = random.choices(range(B + 1), k=1)

for idx in idxs_rand:
    print(namekeys_all[idx])

mm.plot_figure(mgpu['x_Bx3xHxW'][idxs_rand], mgpu['x_Bx2'][idxs_rand], mgpu['y_Bx1xHxW'][idxs_rand])

# # mc_gpu['gpu_x_Bx3xHxW'], mc_gpu['gpu_x_Bx2'], mc_gpu['gpu_y_Bx1xHxW'] = ppcc.load_a_sample(namekey="train_strasbourg_000001_031427_96x422")
# #
# # mm.predict(mc_gpu['gpu_x_Bx3xHxW'], mc_gpu['gpu_x_Bx2'], mc_gpu['gpu_y_Bx1xHxW'])
#
#
# # X, Y = ppcc.load_a_sample('bremen_000273_000019_270x832_static')
# # Y_hat = mm.predict(X, Y, bodynames=['bremen_000273_000019_270x832_static'], bids=[0])
# # print(Y_hat.shape)
"""

    """
    def plot_figure(self, x_Bx3xHxW, x_Bx2, y_Bx1xHxW):

        B, _, H, W = x_Bx3xHxW.shape

        is_training_model = self.model.training

        self.model.eval()
        mgpu = MemCache()

        imgs = []
        titles = []
        with torch.no_grad():
            mgpu['ys_pred_gs_BxKxHSxWS'], mgpu['grid_pred_BxHSxWSx2'], mgpu['xs_pred_gs_Bx3xHSxWS'], mgpu['dm_pred_Bx1xHSxWS'] = self.model(x_Bx3xHxW, x_Bx2)
            _, K, HS, WS = mgpu['ys_pred_gs_BxKxHSxWS'].shape

            mgpu['y_real_BxHSxWS'] = self.model.downsample_y_real_BxHSxWS(y_Bx1xHxW[:, 0, :, :], mgpu['grid_pred_BxHSxWSx2'])

            print(mgpu['ys_pred_gs_BxKxHSxWS'].shape)
            print(mgpu['y_real_BxHSxWS'].shape)

            mgpu['y_pred_BxHxW'] = self.model.output_y_pred_BxHxW(mgpu['ys_pred_gs_BxKxHSxWS'], mgpu['grid_pred_BxHSxWSx2'])

            for bidx in trange(0, B):
                imgs.extend([
                    x_Bx3xHxW[bidx],
                    mgpu['dm_pred_Bx1xHSxWS'][bidx],

                    mgpu['xs_pred_gs_Bx3xHSxWS'][bidx],
                    None,
                    mgpu['ys_pred_gs_BxKxHSxWS'][bidx].argmax(dim=0) < (K - 1),
                    mgpu['y_real_BxHSxWS'][bidx] < (K - 1),

                    mgpu['y_pred_BxHxW'][bidx] < (K - 1),
                    y_Bx1xHxW[bidx] < (K - 1)

                ])

                titles.extend([
                    'x_Bx3xHxW',
                    'densitymap_pred_Bx1xHSxWS',
                    'x_gridsample_Bx3xHSxWS',
                    None,
                    'y_pred_gridsample_BxKxHSxWS',
                    'y_real_gridsample_BxHSxWS',
                    'y_pred_BxHxW',
                    'y_real_BxHxW'

                ])

        self.model.train(is_training_model)

        plt_multi_imgshow(imgs, titles, row_col=(2, 4))
        plt.show(block=True)

    """
