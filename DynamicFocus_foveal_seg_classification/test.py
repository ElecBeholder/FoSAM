def test(cls_pred, x):
    B, K = cls_pred.shape
    _,_, H, W = x.shape
    cls_pred = cls_pred.unsqueeze(-1).unsqueeze(-1) #[20,20,1,1]
    cls_pred = cls_pred.expand(-1, -1, H, W) #[20,20,64,128]
    #print('cls_pred = cls_pred.expand(-1, -1, H, W) shape: ', cls_pred.shape)
    #print('cls_pred[:,0:1,:,:].shape', cls_pred[:,0:1,:,:].shape)
    cls_pred_new = cls_pred.clone()
    cls_pred_new[:,0:1,:,:] = cls_pred[:,0:1,:,:] * x
    #print('cls_pred[:,0:1,:,:] = cls_pred[:,0:1,:,:] * x.cuda()', cls_pred.shape)
    #print('C1 cls pred shape', cls_pred_new.shape)
    #print(x.shape)
    return cls_pred_new