import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda import device
from torchvision.ops import roi_align
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from util.misc import inverse_sigmoid
# from refinebox.models.layers.base_refiner import INF



# 自定义残差块
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# 边界框转换函数 (从中心点格式转换为左上角和右下角格式)
def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# 边界框细化器
class ResBboxRefiner(nn.Module):
    def __init__(self, d_model=64, num_reg_layers=2):
        super(ResBboxRefiner, self).__init__()

        self.roi_layer = ROIPooler(
            output_size=(7,7),
            scales=[1 / (2 ** i) for i in range(1, 4)],
            sampling_ratio=1,
            pooler_type='ROIAlignV2',
        )

        # 定义残差块
        self.res_block = nn.Sequential(
            BottleneckBlock(in_channels=512, bottleneck_channels=64, out_channels=64),
            BottleneckBlock(in_channels=64, bottleneck_channels=64, out_channels=64),
            BottleneckBlock(in_channels=64, bottleneck_channels=64, out_channels=64),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reg_module = None
        self.bboxes_deltas = None
        if num_reg_layers is not None and num_reg_layers > 0:
            reg_module = list()
            for _ in range(num_reg_layers):
                reg_module.append(nn.Linear(d_model, d_model, False))
                reg_module.append(nn.LayerNorm(d_model))
                reg_module.append(nn.ReLU(inplace=True))
            self.reg_module = nn.ModuleList(reg_module)
            self.bboxes_deltas = nn.Linear(d_model, 4)

        # 回归层，用于预测边界框的偏移量
        # self.bboxes_deltas = nn.Linear(64, 4)
        # self.logits_deltas = nn.Linear(64, 20)

    def apply_bboxes_deltas(self, deltas: Tensor, bboxes: Tensor) -> Tensor:
        bboxes = inverse_sigmoid(bboxes) + deltas
        bboxes = bboxes.sigmoid()
        return bboxes

    # def apply_logits_deltas(self, deltas: Tensor, logits: Tensor) -> Tensor:
    #
    #
    #     normed_logits = (logits
    #                      + deltas
    #                      - torch.log(1.0
    #                                  + torch.clamp(logits.exp(), max=INF)
    #                                  + torch.clamp(deltas.exp(), max=INF)))
    #     return normed_logits

    def compute_loss(self, pred_bboxes, target_bboxes):
        """
        计算边界框回归损失

        :param pred_bboxes: 预测的边界框
        :param target_bboxes: 目标边界框
        :return: 损失值
        """
        return nn.functional.mse_loss(pred_bboxes, target_bboxes)

    def forward(self, features, predictions: dict, image_whwh: torch.Tensor):
        """
        前向传播函数，用于对边界框进行细化

        :param features: 特征图 (B, C, H, W)，即来自前一层的卷积特征
        :param predictions: 包含初始预测框的字典
        :param image_whwh: 图像的宽高，用于反归一化
        :param target_bboxes: 目标边界框 (B, N, 4)
        :return: 包含细化后边界框和损失的字典
        """
        if not isinstance(features, list):
            features = features.tensors()

        # 假设使用 GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 定义卷积层并移动到设备
        conv_1x1_512 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1).to(device)
        conv_1x1_1024 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1).to(device)
        conv_1x1_2048 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1).to(device)

        # 对每个 tensor 进行卷积时，确保 tensor 也在相同的设备上
        processed_tensors = []
        for i, tensor in enumerate(features):
            tensor = tensor.to(device)  # 确保 tensor 在同一个设备上

            if tensor.shape[1] == 512:
                processed_tensor = conv_1x1_512(tensor)
            elif tensor.shape[1] == 1024:
                processed_tensor = conv_1x1_1024(tensor)
            elif tensor.shape[1] == 2048:
                processed_tensor = conv_1x1_2048(tensor)
            processed_tensors.append(processed_tensor)

        for i in range(5):
            # 提取预测的初始边界框
            bboxes: torch.Tensor = predictions['pred_boxes']
            bboxes_onetomany: torch.Tensor = predictions['pred_boxes_one2many']

            # 将归一化的边界框转换为像素坐标，并转换为xyxy格式
            unnorm_bboxes = box_cxcywh_to_xyxy(bboxes * image_whwh.unsqueeze(1))
            unnorm_bboxes_onetomany = box_cxcywh_to_xyxy(bboxes_onetomany * image_whwh.unsqueeze(1))

            # 创建Boxes对象，用于后续的ROI提取
            roi_boxes = [Boxes(b) for b in unnorm_bboxes]
            roi_boxes_onetomany = [Boxes(b) for b in unnorm_bboxes_onetomany]

            # 提取ROI特征
            _roi_feats = self.roi_layer(processed_tensors, roi_boxes)
            _roi_feats_onetomany = self.roi_layer(processed_tensors, roi_boxes_onetomany)

            # 通过残差块处理ROI特征
            roi_feats = self.res_block(_roi_feats)
            roi_feats = self.avg_pool(roi_feats).squeeze(-1).squeeze(-1)
            roi_feats_onetomany = self.res_block(_roi_feats_onetomany)
            roi_feats_onetomany = self.avg_pool(roi_feats_onetomany).squeeze(-1).squeeze(-1)

            # # 将ROI特征转换为 (B, N, D) 的格式
            # roi_feats = roi_feats.unflatten(0, bboxes.shape[:2])

            # (B*N, D)
            roi_feats = roi_feats.squeeze(-1).squeeze(-1)
            roi_feats_onetomany = roi_feats_onetomany.squeeze(-1).squeeze(-1)
            # (B*N, D) -> (B, N, D)
            roi_feats = roi_feats.unflatten(0, bboxes.shape[:2])
            roi_feats_onetomany = roi_feats_onetomany.unflatten(0, bboxes_onetomany.shape[:2])
            reg_feature = roi_feats
            reg_feature_onetomany = roi_feats_onetomany

            for reg_layer in self.reg_module:
                reg_feature = reg_layer(reg_feature)
            pred_bboxes_deltas = self.bboxes_deltas(reg_feature)
            pred_bboxes = self.apply_bboxes_deltas(pred_bboxes_deltas, bboxes)

            for reg_layer in self.reg_module:
                reg_feature_onetomany = reg_layer(reg_feature_onetomany)
            pred_bboxes_onetomany_deltas = self.bboxes_deltas(reg_feature_onetomany)
            pred_bboxes_onetomany = self.apply_bboxes_deltas(pred_bboxes_onetomany_deltas, bboxes_onetomany)
            # # 通过回归层预测边界框的调整量
            # pred_bboxes_deltas = self.bboxes_deltas(roi_feats)
            #
            #
            # # 应用边界框的调整量，细化边界框
            # pred_bboxes = self.apply_bboxes_deltas(pred_bboxes_deltas, bboxes)
            predictions['pred_boxes'] = pred_bboxes.clamp(min=0.0, max=1.0)  # 限制边界框在图像内
            predictions['pred_boxes_one2many'] = pred_bboxes_onetomany.clamp(min=0.0, max=1.0)

        return predictions
