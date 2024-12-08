import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device, criterion):
    net.eval()
    x_margin,y_margin = net.getOutput2DMargins()
    val_btch_nb = len(dataloader)
    dice_score = 0
    loss = 0
    # iterate over the validation set
    for batch_idx, batch in enumerate(dataloader):
        image, mask_true = batch['image'], batch['label'][:,x_margin:-1*x_margin,y_margin:-1*y_margin]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        
        with torch.no_grad():
            #calculate loss
            mask_pred = net(image)
            mask_pred = mask_pred.to(device=device, dtype=torch.float32)
            loss +=  criterion(mask_pred,mask_true).item()
            #print('val loss',loss)
            
            
            #calculate dice score
        
            #mask_pred = net(image)
            #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #print("mask pred size",mask_pred.size())
            #print("mask true size",mask_true.size())
            #dice_score += float(multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False))
            
            
           
    #print('total val loss',loss)
    net.train()

    # Fixes a potential division by zero error
    #if val_btch_nb == 0:
    #    return dice_score
    return loss / val_btch_nb , dice_score / val_btch_nb
