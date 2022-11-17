import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.checkpoint import load_clip_to_cpu
from .semantic import semantic
from clip import clip
from .element_wise_layer import Element_Wise_Layer

class SSCLIP(nn.Module):
    def __init__(self, args, word_features, classname,
                image_feature_dim=512, num_classes=80, 
                word_feature_dim=300):
        super(SSCLIP, self).__init__()
        self.clip_model = load_clip_to_cpu(args).float()
        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.word_semantic = semantic(num_classes= self.num_classes,
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)

        self.word_features = self.load_features(word_features)
        # self.text_features = self.get_text_features(classname)
        self.text_features = self.load_text_feature(args.text_file)

        # self.fc_output = nn.Linear(2*self.image_feature_dim, self.output_dim)
        # self.classifiers = Element_Wise_Layer(self.num_classes, self.image_feature_dim)
        self.fc = nn.Linear(self.image_feature_dim, 512)

    def forward(self, image):
        batch_size = image.size()[0]
        feature_size = image.size()[2] // 64
        # encoder attn
        # img_feature_map = self.clip_model.encode_image(image).view(feature_size, feature_size, -1, self.image_feature_dim).permute(2, 3, 0, 1)
        img_feature_map = self.clip_model.encode_image(image.type(self.dtype))  #[bs, H, W, 512]
        # SD
        sd_features = self.word_semantic(batch_size,
                                         img_feature_map,
                                         self.word_features) # [bs, 80, 512]
        sd_features = self.fc(sd_features)
        text_features = self.text_features # [80, 512]
                                     
        sd_features = sd_features / sd_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        output = torch.cosine_similarity(sd_features, text_features, dim=-1) # [bs, 80]

        # contrastive_logits = logit_scale * sd_features @ text_features.t()
        
        return output
    
    def load_features(self,word_features):
        return nn.Parameter(torch.from_numpy(np.load(word_features).astype(np.float32)), requires_grad=False)

    def get_text_features(self, classname):
        token = clip.tokenize(classname).cuda()
        return self.clip_model.encode_text(token)
    
    def load_text_feature(self, text_features):
        return torch.from_numpy(np.load(text_features).astype(np.float32)).cuda()