import torch
import torch.nn as nn
from utils import intersection_over_union



class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S 
        self.B = B 
        self.C = C 
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
        
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5 )
        
        iou_b1 = intersection_over_union(predictions[...,21:25], target[...,21:25])
        iou_b2 = intersection_over_union(predictions[...,26:30], target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)], dim=0)
        
        iou_maxes, bestbox = torch.max(ious,dim=0)
        exist_box = target[...,20].unsqueeze(3)       #bunu aldığımızda 4. boyut kayboluyor. identity object_i'a denk geliyor.
        
        
    ## Box koordinatları için
    
        box_predictions = exist_box * (
            ( bestbox * predictions[...,26:30] + (1 - bestbox) * predictions[...,21:25] )            
            )
        box_targets = exist_box * target[...,21:25]
        
        box_predictions[..., 2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4]+ 1e-6 ))
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])
                                                                           
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
            )
   
    ## Object loss için
    
        pred_box = (
            bestbox * predictions[...,25:26] + (1 - bestbox) * predictions[...,20:21] )          
        
        
        object_loss = self.mse(
            torch.flatten(exist_box*pred_box),
            torch.flatten(exist_box*target[...,20:21]),
            )

    ## No Object loss için

        no_object_loss = self.mse(
            torch.flatten((1-exist_box) * predictions[...,20:21], start_dim=1),
            torch.flatten((1-exist_box) * target[...,20:21], start_dim=1)
            )
        
        no_object_loss += self.mse(
            torch.flatten((1-exist_box) * predictions[...,25:26], start_dim=1),
            torch.flatten((1-exist_box) * target[...,20:21], start_dim=1)
            )
        

    ## Sınıf loss için

        class_loss = self.mse(
            torch.flatten(exist_box * predictions[...,:20], end_dim=-2),
            torch.flatten(exist_box * target[...,:20], end_dim=-2)
            )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss



























































