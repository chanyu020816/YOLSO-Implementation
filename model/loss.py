import torch
import torch.nn as nn
# from model.utils import intersection_over_union

class YOLSOV1Loss(nn.Module):
    def __init__(self, grid_num, num_classes, lambda_coord = 1.5, lambda_size = 1, lambda_class = 1):
        super().__init__()
        self.grid_num = grid_num
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_size = lambda_size
        self.lambda_class = lambda_class

        self.sigmoid = nn.Sigmoid()
        self.l1loss = nn.L1Loss()
        self.cls_loss = nn.CrossEntropyLoss()

    @staticmethod
    def weighted_cross_entropy(pred, target, bg_class_index, bg_weight = 0.1):
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        weights = torch.ones_like(ce_loss)
        class_pred_max = torch.argmax(pred, dim=1)
        weight_mask = (target[:, bg_class_index, :, :] == 0) & (class_pred_max == bg_class_index)
        weights[weight_mask] = bg_weight
        weighted_loss = ce_loss * weights
        return weighted_loss.mean()

    def compute_loss(self, pred, target):
        obj_mask = 1 - target[:, self.num_classes, :, :]
        # total_obj = torch.sum(obj_mask) # total number of objects
        obj_mask = obj_mask.unsqueeze(1).expand(-1, 12, -1, -1)
        # bgobj = target[:, self.num_classes, :, :]

        obj_pred = obj_mask * pred
        obj_target = obj_mask * target

        # ======================== #
        #   FOR COORDINATES LOSS   #
        # ======================== #
        coord_pred = obj_pred[:, self.num_classes+1:self.num_classes+3, :, :]
        coord_target = obj_target[:, self.num_classes+1:self.num_classes+3, :, :]
        coord_loss = self.l1loss(coord_pred, coord_target)

        # ======================== #
        #  FOR RELATIVE SIZE LOSS  #
        # ======================== #
        size_pred = obj_pred[:, self.num_classes+3, :, :]
        size_target = obj_target[:, self.num_classes+3, :, :]
        size_loss = self.l1loss(size_pred, size_target)

        # ======================== #
        #      FOR CLASS LOSS      #
        # ======================== #
        class_pred = obj_pred[:, :self.num_classes+1, :, :]
        class_target = obj_target[:, :self.num_classes+1, :, :]
        class_loss = self.weighted_cross_entropy(class_pred, class_target, self.num_classes)

        # ======================== #
        #     FOR OVERALL LOSS     #
        # ======================== #
        total_loss = (
            self.lambda_coord * coord_loss +
            self.lambda_size * size_loss +
            self.lambda_class * class_loss
        )
        return total_loss, coord_loss, size_loss, class_loss

    def forward(self, pred, target):
        total_loss, coord_loss, size_loss, class_loss = self.compute_loss(pred, target)
        return total_loss

class YOLSOV1ValLoss(YOLSOV1Loss):
    def __init__(self, grid_num, num_classes):
        super().__init__(grid_num, num_classes)

    def forward(self, pred, target):
        total_loss, coord_loss, size_loss, class_loss = self.compute_loss(pred, target)
        return total_loss, coord_loss, size_loss, class_loss

if __name__ == '__main__':
    loss = YOLSOV1Loss(grid_num=3, num_classes=2)
    print(loss(10, torch.randn(10, 10, 10)))