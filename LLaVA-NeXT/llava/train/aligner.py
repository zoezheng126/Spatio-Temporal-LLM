import torch
import torch.nn as nn
import importlib

from typing import List, Optional, Tuple, Union, Dict
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.train.ray_encoder import RayDirectionEncoderForImage,RayDirectionEncoderForPointCloud
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig, LlavaQwenModel
import transformers

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, InstructBlipQFormerModel, InstructBlipQFormerConfig
from torch.cuda.amp import autocast

import deepspeed
import torch.distributed as dist

from llava.model.openscene.openscene_inference import load_cfg_from_cfg_file
from llava.model.openscene.models.disnet import DisNet
from MinkowskiEngine import SparseTensor

from llava.model.openscene.third_party.pointnet2 import pointnet2_utils  
import torch.nn.functional as F


def scale_intrinsics(K, H_orig, W_orig, H, W):
    scale_x = W / W_orig
    scale_y = H / H_orig
    K_scaled = K
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled

class LlavaQwenPcdAlignerConfig(LlavaQwenConfig):
    model_type = "pcd_aligner"
    
    def __init__(self,
                 use_color=True,
                 use_normal=True,
                 use_height=True,
                 use_multiview=False,
                 pcd_aligned_method="linear",
                 pcd_view_base_pe=127,
                 pcd_n_points=1024,
                 pcd_modality="pointcloud",
                 detector_model_name="Model_Vote2Cap_DETR",
                 detector_pcd_feature_dim=256,
                 img_encoder_feature_dim=3584,
                 img_aligned_method="linear",
                 img_view_base_pe=127,
                 img_feature_dim=768,
                 img_modality="video",
                 img_num_patches = 196,
                 decoder_layer_d_model=512,  
                 decoder_layer_nhead=4,
                 decoder_layer_dim_feedforward=256,
                 decoder_layer_dropout=0.1,
                 decoder_num_layers=4,
                 in_dim_proj = 3584,
                 out_dim_proj = 256,
                 **kwargs):
        super().__init__(**kwargs)

        self.model_type = "pcd_aligner"
        self.architectures = ["LlavaQwenPcdVidLM"]
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_height = use_height
        self.use_multiview = use_multiview

        self.pcd_aligned_method = pcd_aligned_method  # ray encoder
        self.pcd_view_base_pe = pcd_view_base_pe # ray encoder
        self.pcd_feature_dim = pcd_feature_dim # ray encoder
        self.pcd_n_points = pcd_n_points # ray encoder
        self.pcd_modality = pcd_modality # ray encoder
        self.detector_model_name = detector_model_name # pcd encoder
        self.img_encoder_feature_dim = img_encoder_feature_dim
        self.img_aligned_method = img_aligned_method # ray encoder
        self.img_view_base_pe = img_view_base_pe  # ray encoder
        self.img_feature_dim = img_feature_dim   # ray encoder
        self.img_modality = img_modality #ray encoder
        self.img_num_patches = img_num_patches # ray encoder
        self.decoder_layer_d_model = decoder_layer_d_model  # pcd-img aligner decoder
        self.decoder_layer_nhead = decoder_layer_nhead
        self.decoder_layer_dim_feedforward = decoder_layer_dim_feedforward
        self.decoder_layer_dropout = decoder_layer_dropout
        self.decoder_num_layers = decoder_num_layers
        self.detector_pcd_feature_dim = detector_pcd_feature_dim

        self.in_dim_proj = in_dim_proj  # LLM hidden dimension
        self.out_dim_proj = out_dim_proj  

    def get_module_config(self, module='decoder'):
        return {
            key.replace(f"{module}_", ""): value  
            for key, value in vars(self).items()  
            if key.startswith(f"{module}_") 
        }


class LlavaQwenPcdVidLM(LlavaQwenForCausalLM):
    config_class = LlavaQwenPcdAlignerConfig 
    _keep_in_fp32_modules = ["pcd_encoder"]
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.pcd_ray_encoder = RayDirectionEncoderForPointCloud(**config.get_module_config("pcd"))
        self.img_ray_encoder = RayDirectionEncoderForImage(**config.get_module_config("img"))

        cfg = load_cfg_from_cfg_file('LLaVA-NeXT/llava/model/openscene/config/scannet/ours_openseg_pretrained.yaml')
        self.pcd_encoder = DisNet(cfg)

        in_channel, enc_dim = 768, 768
        mlp_dim = mlp_spec = [in_channel, 64, 128, enc_dim]

        self.radius = 0.2
        self.nsample = 64
        self.use_xyz = False 
        self.normalize_xyz = False 
        self.sample_uniformly = False
        self.ret_unique_cnt = False
        self.bn = True
        self.npoint=1024
        self.grouper = pointnet2_utils.QueryAndGroup(
                self.radius, self.nsample, use_xyz=self.use_xyz,
                ret_grouped_xyz=True, normalize_xyz=self.normalize_xyz,
                sample_uniformly=self.sample_uniformly, ret_unique_cnt=self.ret_unique_cnt)
        
        layers = []
        for i in range(len(mlp_spec) - 1):
            layers.append(nn.Linear(mlp_spec[i], mlp_spec[i+1]))
            if self.bn:
                layers.append(nn.BatchNorm1d(mlp_spec[i+1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp_module = nn.Sequential(*layers)

        qformer_config = InstructBlipQFormerConfig(
            num_hidden_layers=6,
            encoder_hidden_size=768 
        )

        self.qformer = InstructBlipQFormerModel(config=qformer_config)

        self.qformer_hidden_size = qformer_config.hidden_size
        self.nlatent_query = 32
        self.img_encoder_to_qformer_projection = nn.Linear(config.img_encoder_feature_dim, self.qformer_hidden_size)
        self.text_embed_to_qformer_projection = nn.Linear(config.img_encoder_feature_dim, self.qformer_hidden_size)
        self.latent_query = nn.Embedding(self.nlatent_query, self.qformer_hidden_size)
        self.qformer_to_language_projection = nn.Linear(self.qformer_hidden_size, config.in_dim_proj)

        self.config = config
        self.pcd_encoder_config = cfg
        self.post_init()

    def _init_weights(self, module):
        # Norm layers
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Embedding layers
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

        # Linear layers
        elif isinstance(module, nn.Linear):
            if module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Conv layers
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Other modules with parameters
        else:
            for name, param in module.named_parameters(recurse=False):
                if param is None:
                    continue
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'norm' in name:
                    nn.init.ones_(param)
                else:
                    nn.init.zeros_(param)

    def image_preprocess_for_alignment(self, 
                                       images = None, 
                                       img_pos = None):
        _, C, H, W = images[0].shape
        bs = len(images) if isinstance(images, list) else images.shape[0]
        flatten_images_tensor = torch.cat(images, dim=0).view(-1, C, H, W)
        flatten_img_features = self.encode_images(flatten_images_tensor)
        num_frames_list = [image.shape[0] for image in images]
        img_features = list(torch.split(flatten_img_features, num_frames_list))
        img_features = [self.img_encoder_to_qformer_projection(self.get_2dPool(img_feature)) for img_feature in img_features]

        img_features = [img_feature.flatten(0,1) for img_feature in img_features] # item shape:  num_frames * num_patches, feature_dim
        img_pos = [pos.flatten(0,1) for pos in img_pos]
        max_num_patches = max(feature.shape[0] for feature in img_features)
        batch_size = len(img_features)
        feature_dim = img_features[0].shape[-1]

        padded_features = torch.zeros(batch_size, max_num_patches, feature_dim, dtype=img_features[0].dtype, device=img_features[0].device)
        padded_pos = torch.zeros(batch_size, max_num_patches, feature_dim, dtype=img_features[0].dtype, device=img_features[0].device)
        attention_mask = torch.zeros(batch_size, max_num_patches, dtype=torch.bool, device=img_features[0].device)

        for i, (feature, pos) in enumerate(zip(img_features, img_pos)):
            num_patches = feature.shape[0]
            padded_features[i, :num_patches, :] = feature  # Copy data
            padded_pos[i, :num_patches, :] = pos  # Copy data
            attention_mask[i, :num_patches] = True

        return img_features, padded_pos, padded_features, attention_mask
    
    def _get_instruction_response(self, 
            pcd_feature_with_pos: torch.Tensor,
            img_feature_with_pos: torch.Tensor,
            img_attention_mask: torch.Tensor,
            text_attention_mask: torch.Tensor,
            input_ids: torch.Tensor,
            batch_size: int,
        ) -> dict:
    
        ## prompt encoding
        valid_mask = (input_ids != -300) & (input_ids != -200)
        input_ids = input_ids[valid_mask].unsqueeze(0)
        text_attention_mask = text_attention_mask[valid_mask].unsqueeze(0)

        text_embeddings = self.get_model().embed_tokens(input_ids)
        text_embeddings = self.text_embed_to_qformer_projection(text_embeddings)

        query_tokens = self.latent_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_tokens = torch.cat((query_tokens, img_feature_with_pos, text_embeddings), dim=1)
        query_attention_mask = torch.cat(
            (torch.ones(batch_size, self.nlatent_query, dtype=img_attention_mask.dtype, device=img_attention_mask.device),
            img_attention_mask, text_attention_mask),
            dim=1
        )

        query_outputs = self.qformer(
            input_ids=None,
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=pcd_feature_with_pos,
        )
        query_outputs = query_outputs[0][:, : self.nlatent_query, :]
    
        prefix_feature = self.qformer_to_language_projection(query_outputs)
        
        return prefix_feature

    def safe_encode_pointcloud(self,
                               xyz,
                                rgb,
                                batch_idx=None,
                                npoint=2048):
        """
        Robust wrapper around self.pcd_encoder().

        Args
        ----
        xyz  : (N, 3) world-space coordinates
        rgb  : (N, 3) colours ∈ [0,1]
        batch_idx : (N,) batch indices if you already have them.
                    If None, assume a single sample (all zeros).

        Returns
        -------
        pcd_features, pcd_xyz - whatever self.pcd_encoder normally returns.
        """

        if batch_idx is None:
            batch_idx = torch.ones_like(xyz[:, :1], dtype=torch.float32)

        def build_sparse(xyz_, rgb_):
            # Minkowski uses (N, 4) integer coords: (b, x, y, z)
            coords = torch.cat([batch_idx, xyz_], dim=1).float()
            feats  = rgb_.float()             # (N, 3)
            return SparseTensor(feats, coords)

        sinput = build_sparse(xyz, rgb)
        self.pcd_encoder.eval()
        pcd_features, output_xyz = self.pcd_encoder(sinput)
        return pcd_features, output_xyz
        try:
            self.pcd_encoder.eval()
            pcd_features, output_xyz = self.pcd_encoder(sinput)
            return pcd_features,output_xyz                 # ------ 1st try
        except Exception as e:
        
            # ---------- FALLBACK: down‑sample and retry ----------
            n_pts_in = xyz.shape[0]
            if n_pts_in < npoint:
                # Repeat points to reach npoint length
                num_pad = npoint - n_pts_in
                pad_inds = torch.randint(0, n_pts_in, (num_pad,), device=xyz.device)
                
                xyz = torch.cat([xyz, xyz[pad_inds]], dim=0)      # (npoint, 3)
                rgb = torch.cat([rgb, rgb[pad_inds]], dim=0)      # (npoint, C)
                n_pts_in = xyz.shape[0]  # should now be npoint


            # FPS expects (B, N, 3); here B = 1
            xyz_batched = xyz.unsqueeze(0)          # (1, N, 3)
            fps_inds = pointnet2_utils.furthest_point_sample(
                xyz_batched, npoint).squeeze(0)           # (npoint,)

            xyz_ds = xyz[fps_inds]                 # (npoint, 3)
            rgb_ds = rgb[fps_inds]

            sinput_ds = build_sparse(xyz_ds, rgb_ds)

            # second (and final) attempt
            return self.pcd_encoder(sinput_ds)
        
    def forward(
        self,
        pointclouds: Optional[torch.Tensor] = None,
        pointclouds_feature: Optional[torch.Tensor] = None,
        pointclouds_colors: Optional[torch.Tensor] = None,
        pointclouds_labels: Optional[torch.Tensor] = None,
        pointclouds_xyz: Optional[torch.Tensor] = None,
        pointclouds_attn_mask: Optional[List[bool]] = None,
        K: Optional[torch.Tensor] = None,
        c2ws: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.Tensor] = None,
        qformer_labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        pointcloud_modalities: Optional[List[str]] = None,
        pointcloud_sizes: Optional[List[int]] = None,
        dpo_forward: Optional[bool] = False,
        target_bbox_corner: Optional[torch.Tensor] = None,
        target_bbox_center: Optional[torch.Tensor] = None,
        target_object_id: Optional[torch.Tensor] = None,
        target_object_name: Optional[torch.Tensor] = None,
        qa_info: Optional[List[Dict]] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        pcd_xyz_list = []
        pcd_features_list = []

        if (inputs_embeds is None and pointclouds is not None and input_ids.shape[1] != 1):
            no_pcd_in_batch = all(mask is False for mask in pointclouds_attn_mask)   
            if (pointclouds is not None or pointclouds_feature is not None) and no_pcd_in_batch is False:
                pointclouds_tensor = torch.cat(pointclouds, dim=0).to(torch.float32)
                pointclouds_colors_tensor = torch.cat(pointclouds_colors, dim=0).to(torch.float32)
                
                pcd_features, pcd_xyz = self.safe_encode_pointcloud(xyz=pointclouds_tensor,rgb=pointclouds_colors_tensor)  # ! change back when handle bn error
                pcd_features = pcd_features.unsqueeze(0).to(torch.bfloat16)  # ! hardcoded to batch=1 
                pcd_xyz_list = [pcd_xyz[:, 1:].unsqueeze(0)]    # ! TODO hardcoded to batch=1
                viewdirs_pcds = []
                for cur_idx in range(len(pcd_xyz_list)):
                    rays_pcd_o, rays_pcd_d, viewdirs_pcd = self.pcd_ray_encoder.compute_rays_for_pointcloud_batch(pcd=pcd_xyz_list[cur_idx], c2ws=c2ws[cur_idx].unsqueeze(0))
                    viewdirs_pcd = viewdirs_pcd.to(torch.bfloat16)   
                    viewdirs_pcds.append(viewdirs_pcd)
               
                viewdirs_pcds = torch.cat(viewdirs_pcds, dim=0)
                self.pcd_ray_encoder.n_points = pcd_features.shape[1] 
                pcd_pos = self.pcd_ray_encoder(viewdirs_pcd) #.permute(1,0,2)  # bs, n_points, feature_dim
                pcd_features_with_pos = pcd_features + pcd_pos

# image
                viewdirs_imgs = []
                img_pos = []
                for cur_idx in range(len(images)):
                    _, _, H, W = images[cur_idx].shape
                
                    _, H_orig, W_orig, _ = image_sizes[cur_idx]
                    K[cur_idx] = scale_intrinsics(K[cur_idx], H_orig, W_orig, H, W)
                    
                    rays_img_o, rays_img_d, viewdirs_img = self.img_ray_encoder.compute_rays_for_video_batch(H=H, W=W, K=K[cur_idx], c2ws=c2ws[cur_idx].unsqueeze(0)) # ! zhz hardcoded remove float() later
                    viewdirs_img = viewdirs_img.to(torch.bfloat16)
                    pos = self.img_ray_encoder(viewdirs_img.unsqueeze(0), H, W, is_normalized=True).squeeze(0) 
                    img_pos.append(pos)
                    viewdirs_imgs.append(viewdirs_img)

                img_features, padded_img_pos, padded_img_features, img_attention_mask = self.image_preprocess_for_alignment(images, img_pos)
                img_feature_with_pos = padded_img_features + padded_img_pos


                xyz = pcd_xyz_list[0].float()
                xyz_flipped = xyz.transpose(1, 2).contiguous().float()
                inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)  # ! not control
                new_xyz = pointnet2_utils.gather_operation(  # ! not control
                    xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
               
                device = pcd_features_with_pos.device 
             
                img_feature_with_pos = img_feature_with_pos.to(torch.bfloat16)
                pcd_features_with_pos = pcd_features_with_pos.to(torch.bfloat16)  #! change back
               
                prefix_tokens = self._get_instruction_response(
                    pcd_feature_with_pos=pcd_features_with_pos,
                    img_feature_with_pos=img_feature_with_pos,
                    img_attention_mask=img_attention_mask,
                    text_attention_mask=qformer_attention_mask,
                    input_ids=qformer_input_ids,
                    batch_size=img_feature_with_pos.shape[0],
                )
             
                prefix_tokens = [prefix_tokens[i,:, :] for i in range(prefix_tokens.shape[0])]

            else:
                prefix_tokens = None

            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_3d_multimodal(input_ids, 
                                                                                                                                            position_ids, 
                                                                                                                                            attention_mask, 
                                                                                                                                            past_key_values, 
                                                                                                                                            labels, 
                                                                                                                                            images, 
                                                                                                                                            modalities, 
                                                                                                                                            image_sizes,
                                                                                                                                            pointcloud_feature=prefix_tokens)
            
        elif inputs_embeds is None and input_ids.shape[-1] == 1:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
    
    @torch.no_grad()
    def generate(
        self,
        pointclouds: Optional[torch.Tensor] = None,
        pointclouds_feature: Optional[torch.Tensor] = None,
        pointclouds_xyz: Optional[torch.Tensor] = None,
        pointclouds_colors: Optional[torch.Tensor] = None,
        pointclouds_attn_mask: Optional[List[bool]] = None,
        K: Optional[torch.Tensor] = None,
        c2ws: Optional[torch.Tensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.Tensor] = None,
        qformer_labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        pointcloud_modalities: Optional[List[str]] = None,
        pointcloud_sizes: Optional[List[int]] = None,
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        qa_info: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop('position_ids', None)
        attention_mask = kwargs.pop('attention_mask', None)
        inputs_embeds = kwargs.pop('inputs_embeds', None)
        if inputs_embeds is not None:
            return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

        print('667 input_ids: {input_ids}')
        if (inputs_embeds is None and pointclouds is not None and input_ids.shape[1] != 1):
            no_pcd_in_batch = all(mask is False for mask in pointclouds_attn_mask)
            if (pointclouds is not None or pointclouds_feature is not None) and no_pcd_in_batch is False:
                pointclouds_tensor = torch.cat(pointclouds, dim=0).to(torch.float32)
                pointclouds_colors_tensor = torch.cat(pointclouds_colors, dim=0).to(torch.float32)
            
                pcd_features, pcd_xyz = self.safe_encode_pointcloud(xyz=pointclouds_tensor,rgb=pointclouds_colors_tensor)
                pcd_features = pcd_features.unsqueeze(0).to(torch.bfloat16)

                pcd_xyz_list = [pcd_xyz[:, 1:].unsqueeze(0)]   
                viewdirs_pcds = []
                for cur_idx in range(len(pcd_xyz_list)):
                    rays_pcd_o, rays_pcd_d, viewdirs_pcd = self.pcd_ray_encoder.compute_rays_for_pointcloud_batch(pcd=pcd_xyz_list[cur_idx], c2ws=c2ws[cur_idx].unsqueeze(0))
                    viewdirs_pcd = viewdirs_pcd.to(torch.bfloat16)   
                    viewdirs_pcds.append(viewdirs_pcd)
               
                viewdirs_pcds = torch.cat(viewdirs_pcds, dim=0)
                self.pcd_ray_encoder.n_points = pcd_features.shape[1]
                pcd_pos = self.pcd_ray_encoder(viewdirs_pcd) #.permute(1,0,2)  # bs, n_points, feature_dim
                pcd_features_with_pos = pcd_features + pcd_pos

# image

                viewdirs_imgs = []
                img_pos = []
                for cur_idx in range(len(images)):
                    _, _, H, W = images[cur_idx].shape
                
                    _, H_orig, W_orig, _ = image_sizes[cur_idx]
                    K[cur_idx] = scale_intrinsics(K[cur_idx], H_orig, W_orig, H, W)
                    
                    rays_img_o, rays_img_d, viewdirs_img = self.img_ray_encoder.compute_rays_for_video_batch(H=H, W=W, K=K[cur_idx], c2ws=c2ws[cur_idx].unsqueeze(0))
                    
                    viewdirs_img = viewdirs_img.to(torch.bfloat16)
                    pos = self.img_ray_encoder(viewdirs_img.unsqueeze(0), H, W, is_normalized=True).squeeze(0) 
                    img_pos.append(pos)
                    viewdirs_imgs.append(viewdirs_img)

                img_features, padded_img_pos, padded_img_features, img_attention_mask = self.image_preprocess_for_alignment(images, img_pos)
                img_feature_with_pos = padded_img_features + padded_img_pos

                img_feature_with_pos = img_feature_with_pos.to(torch.bfloat16)
                pcd_features_with_pos = pcd_features_with_pos.to(torch.bfloat16)
               
                prefix_tokens = self._get_instruction_response(
                    pcd_feature_with_pos=pcd_features_with_pos,
                    img_feature_with_pos=img_feature_with_pos,
                    img_attention_mask=img_attention_mask,
                    text_attention_mask=qformer_attention_mask,
                    input_ids=qformer_input_ids,
                    batch_size=img_feature_with_pos.shape[0],
                )
             
                prefix_tokens = [prefix_tokens[i,:, :] for i in range(prefix_tokens.shape[0])]

            else:
                prefix_tokens = None

            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_3d_multimodal(input_ids, 
                                                                                                                                            position_ids, 
                                                                                                                                            attention_mask, 
                                                                                                                                            past_key_values, 
                                                                                                                                            labels, 
                                                                                                                                            images, 
                                                                                                                                            modalities, 
                                                                                                                                            image_sizes,
                                                                                                                                            pointcloud_feature=prefix_tokens)
            
            return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        
    
AutoConfig.register("pcd_aligner", LlavaQwenPcdAlignerConfig)
AutoModelForCausalLM.register(LlavaQwenPcdAlignerConfig, LlavaQwenPcdAlignerConfig)

