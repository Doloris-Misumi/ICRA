import torch
import torch.nn as nn
import torch.nn.functional as F

from models import pre_processor, backbone_3d, head, roi_head, img_cls

class RL3DF_gate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL
        
        self.weather_vocab = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']
        prompt_cfg = self.cfg_model.get('CONDITION_PROMPT', {})
        self.prompt_source = str(prompt_cfg.get('SOURCE', 'image_probs')).lower()
        self.use_text_prompt = bool(prompt_cfg.get('USE_TEXT_PROMPT', False))
        self.use_gt_prompt_in_train = bool(prompt_cfg.get('USE_GT_PROMPT_IN_TRAIN', False))
        self.allow_gt_prompt_at_test = bool(prompt_cfg.get('ALLOW_GT_PROMPT_AT_TEST', False))
        self.prompt_temperature = max(float(prompt_cfg.get('TEMPERATURE', 1.0)), 1e-6)
        self.enable_contrastive = bool(self.cfg_model.get('ENABLE_CONTRASTIVE', False))
        self.text_encoder = None
        if self.enable_contrastive:
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # ln(14.28)
        else:
            self.register_parameter('logit_scale', None)
        if self.use_text_prompt or self.enable_contrastive:
            from models.text_encoder.clip_encoder import TextEncoder
            self.text_encoder = TextEncoder(freeze=True)
            self.text_encoder.eval()
            weather_prompts = [f"A {w} driving scene" for w in self.weather_vocab]
            with torch.no_grad():
                weather_features = self.text_encoder(weather_prompts)
                weather_features = F.normalize(weather_features, dim=-1)
        else:
            weather_features = torch.zeros(len(self.weather_vocab), 512)
        self.register_buffer('weather_features', weather_features)
        if self.use_text_prompt:
            self.prompt_token_proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU()
            )
        else:
            self.prompt_token_proj = None
        
        self.list_module_names = [
            'pre_processor', 'pre_processor2', 'img_cls', 'backbone_3d', 'head', 'roi_head', 
        ]
        self.list_modules = []
        self.build_rl_detector()

    def _allow_metadata_prompt(self):
        if self.training:
            return self.use_gt_prompt_in_train
        return self.allow_gt_prompt_at_test

    def _build_prompt_weather_token(self, x):
        weather_probs = None
        if 'img_cls_output' in x:
            weather_logits = x['img_cls_output']
            if weather_logits.ndim == 1:
                weather_logits = weather_logits.unsqueeze(0)
            weather_probs = torch.softmax(
                weather_logits.detach() / self.prompt_temperature,
                dim=-1
            )
            x['weather_probs'] = weather_probs

        use_metadata_prompt = (
            self.prompt_source in ['gt_prompt', 'metadata_prompt', 'condition_prompt']
            and self.use_text_prompt
            and self._allow_metadata_prompt()
            and 'condition_prompts' in x
            and self.text_encoder is not None
        )

        if use_metadata_prompt:
            with torch.no_grad():
                prompt_features = self.text_encoder(x['condition_prompts'])
                prompt_features = F.normalize(prompt_features, dim=-1)
                weather_logits = prompt_features @ self.weather_features.t()
                weather_probs = torch.softmax(weather_logits, dim=-1)
            x['weather_probs'] = weather_probs
        elif weather_probs is None:
            if 'img_embedding' in x:
                batch_size = x['img_embedding'].shape[0]
                device = x['img_embedding'].device
            else:
                batch_size = int(x.get('batch_size', 1))
                device = self.weather_features.device
            weather_probs = torch.full(
                (batch_size, len(self.weather_vocab)),
                1.0 / len(self.weather_vocab),
                device=device,
                dtype=self.weather_features.dtype,
            )
            x['weather_probs'] = weather_probs

        if self.use_text_prompt:
            weather_features = self.weather_features.to(weather_probs.device)
            prompt_weather_token = weather_probs @ weather_features
            x['prompt_weather_token'] = self.prompt_token_proj(prompt_weather_token)
        else:
            x.pop('prompt_weather_token', None)
        return x

    def build_rl_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_img_cls(self):
        if self.cfg_model.get('IMG_CLS', None) is None:
            return None
        
        module = img_cls.__all__[self.cfg_model.IMG_CLS.NAME]()
        return module 

    def build_pre_processor(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
        return module 
    
    def build_pre_processor2(self):
        if self.cfg_model.get('PRE_PROCESSOR2', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR2.NAME](self.cfg)
        return module 

    def build_backbone_3d(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def train(self, mode=True):
        """Keep the completely frozen stage-1 camera encoder in eval mode."""
        super().train(mode)
        if hasattr(self, 'img_cls') and not any(
            parameter.requires_grad for parameter in self.img_cls.parameters()
        ):
            self.img_cls.eval()
        return self

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module

    def forward(self, x):
        img_cls_module = getattr(self, 'img_cls', None)
        for module in self.list_modules:
            x = module(x)
            if module is img_cls_module:
                x = self._build_prompt_weather_token(x)

        if self.enable_contrastive and self.training and 'condition_prompts' in x and self.text_encoder is not None:
            if 'img_embedding' in x:
                condition_token = x['img_embedding']
            else:
                pass 
            
            if 'img_embedding' in x:
                with torch.no_grad():
                    text_features = self.text_encoder(x['condition_prompts'])
                
                condition_token = F.normalize(condition_token, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                logit_scale = self.logit_scale.exp()
                logits_per_image = logit_scale * condition_token @ text_features.t()
                logits_per_text = logits_per_image.t()
                
                batch_size = condition_token.shape[0]
                labels = torch.arange(batch_size, device=condition_token.device)
                
                loss_i2t = F.cross_entropy(logits_per_image, labels)
                loss_t2i = F.cross_entropy(logits_per_text, labels)
                contrastive_loss = (loss_i2t + loss_t2i) / 2
                
                lambda_contrastive = 0.1 
                x['contrastive_loss'] = lambda_contrastive * contrastive_loss
                if 'logging' not in x: x['logging'] = {}
                x['logging']['loss_contrastive'] = contrastive_loss.item()

        return x
