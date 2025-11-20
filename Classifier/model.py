import torch.nn as nn
from transformers import CLIPVisionModel

class CLIPClassifier(nn.Module):
    def __init__(self, model_name, num_classes=10, dropout_rate=0.1):
        super(CLIPClassifier, self).__init__()
        # 加载预训练的CLIP Vision Tower
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        
        # 获取输出维度 (ViT-Large通常是1024, Base是768)
        self.hidden_size = self.backbone.config.hidden_size
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, num_classes)
        )
        
        # 显式初始化分类头
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # CLIP Vision Model 输出是一个对象，pooler_output 对应 [CLS] token
        outputs = self.backbone(pixel_values=x)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)       #分类头
        return logits