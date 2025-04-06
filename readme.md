# Keep Alignment with Latest Code

No need to generate the new dataset, just update these file with updates.

- **e/a_preprocess_tools_parallel.py**: Update the dataloader.
- **e/b2_preprocess_lvis.py**: Update the dataloader.
- **b/nn_A2_loss.py**: Loss function: BMSE Loss + Dice Loss + TV Loss.
- **b/nn_A3_metrics.py**
- **b/nn_B10.py**: Segformer model
- **b/nn_B11.py**: Deeplab model
- **b/nn_B14.py**: PSPNet model
- **b/nn_B15.py**: HRNet model
- **b/nn_D3.py** 
- **b/nn_D8.py**
- **b/nn_D9.py** 
- **b/nn_D10.py**
- **b/nn_D11.py** 
- **b/nn_E_manager_parallel_stage.py**: Training method, new module, comprehensive report. 

# Running command
- Evaluate the average deeplab performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100 --seg_module deeplab
```

- Evaluate the average seg_former_b4 performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100 --seg_module segformer_4
```
- Evaluate the average segformer_b5 performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100 --seg_module segformer_5
```
- Evaluate the average pspnet performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100 --seg_module pspnet
```
- Evaluate the average hrnet performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100 --seg_module hrnet
```
- Evaluate the saliency deeplab performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerZoomDeep_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100
```
- Evaluate the saliency pspnet performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerZoomPSP_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100
```
- Evaluate the saliency hrnet performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerZoomHR_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100
```

- Evaluate the saliency segformer_b4 performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerZoomSeg_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100 --segformer_depth 4
```
- Evaluate the saliency segformer_b5 performance
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python d_model/nn_E_manager_parallel_stage.py --train --metrics --Module_Loss SegerZoomPSP_BMSELoss --dataset_marker_train sp100000 --dataset_marker_valid sp20000 --downsample_factor 8 --downsample_factor_deformation 8 --batch_size_train 500 --batch_size_valid 100 --class_num 100 --segformer_depth 4
```


## d/nn_E_manager_parallel_stage.py
```
class ModelManager:

    def __init__(self, model, model_name, device, refresh=False):
        self.device = device

        self.model = model
        self.model_name = model_name

        self.fpath_training_records = os.path.join(preset.dpath_training_records, f"{self.model_name}")
        os.makedirs(self.fpath_training_records, exist_ok=True)

        self.recorder = SummaryWriter(self.fpath_training_records)

        self.fpath_work_pt = os.path.join(self.fpath_training_records, f'params.pt')
        self.fpath_view_pth = os.path.join(self.fpath_training_records, f'params.pth')
        self.fpath_msg_json = os.path.join(self.fpath_training_records, f'msg.json')

        loaded = False
        if not refresh:
            print(f'\nload NN d_model state from {self.fpath_work_pt}')
            loaded = self.load_model(self.fpath_work_pt)
        elif not loaded:
            print('\nload NN d_model state fail ; init state')
            if 'SegerZoomSeg' not in model_name and 'SegerZoomDeep' not in model_name and 'SegerZoomPSP' not in model_name and 'SegerZoomHR' not in model_name and 'SegerAverage' not in model_name:
                self.init_model()
            else:
                self.model.module.gen_cls.apply(init_weights_random)
```

## nn_D3_seger_average.py
```
class SegerAverage(nn.Module):
    def __init__(self, base_module: Type[nn.Module], classify_module: Type[nn.Module], class_num=2, in_channels=4, out_channels=1, downsample_factor=16, cbase_seg=32, nlayer_seg=4, seg_module=''):
        super(SegerAverage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = out_channels

        self.downsample_factor = downsample_factor
        
        if seg_module == 'deeplab':
            self.gen_seg = CustomDeepLab(num_input_channels=in_channels+2)
        elif seg_module == 'segformer_4':
            config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
            config.num_labels = 1  
            self.gen_seg = CustomSegformer(config=config, num_input=in_channels+2)
        elif seg_module == 'segformer_5':
            config = SegformerConfig.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            config.num_labels = 1  
            self.gen_seg = CustomSegformer(config=config, num_input=in_channels+2)
        elif seg_module == 'pspnet':
            self.gen_seg = CustomPSPNet(input_channels=in_channels+2, num_classes=out_channels)
        elif seg_module == 'hrnet':
            self.gen_seg = CustomHRNet(in_channels=in_channels+2,num_classes=out_channels)
```