import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model

# Merged #
#from .builder import build_backbone, build_criterion
#from ..structures import Point

from .backbones.reno.network import Network as RenoDecoder
from torchsparse import SparseTensor

import os
# End Merged #

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

        
      
@MODELS.register_module()
class MergedSegmentor(nn.Module):
    def __init__(
        self,
        backbone=None,          # Config dict for PTv3
        criteria=None,          # List of loss configs
        num_classes=8,
        backbone_out_channels=64,
        reno_ckpt=None,         # Path to RENO checkpoint
        ptv3_ckpt=None,         # Path to PTv3 checkpoint
        recon_loss_weight=0.1,  # Weight for the reconstruction loss
    ):
        super().__init__()
        # ======================= 1. BUILD MODELS =======================
        # Build the RENO decoder
        # These parameters are from the original RENO architecture
        self.decoder = RenoDecoder(channels=32, kernel_size=3)
        
        # Build the PTv3 backbone using the provided config dict
        self.segmentation_backbone = build_model(backbone)
        
        # Build the final linear layer for segmentation predictions
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        
        # Build the loss functions from the config
        self.criteria = build_criteria(criteria)
        self.recon_loss_weight = recon_loss_weight
        self.CUDA = True

        # ======================= 2. LOAD CHECKPOINTS =======================
        print("Initializing MergedSegmentor...")
        if reno_ckpt and os.path.exists(reno_ckpt):
            print(f"Loading RENO checkpoint from: {reno_ckpt}")
            self.decoder.load_state_dict(torch.load(reno_ckpt, map_location='cpu'), strict=False)
            
            # FREEZING RENO DECODER WEIGHTS
            print("Freezing RENO decoder weights.")
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            print("WARNING: RENO checkpoint not found. Using random RENO weights.")

        if ptv3_ckpt and os.path.exists(ptv3_ckpt):
            print(f"Loading PTv3 checkpoint from: {ptv3_ckpt}")
            state_dict = torch.load(ptv3_ckpt, map_location='cpu')
            # Pointcept saves optimizer state, etc. The model weights are in the "model" key.
            # strict=False ignores keys that don't match (like a different seg_head)
            self.segmentation_backbone.load_state_dict(state_dict["model"], strict=False)
        else:
            print("WARNING: PTv3 checkpoint not found. Using random PTv3 weights.")

            
    def forward(self, input_dict):
            if self.CUDA:
                device = "cuda:0"
            else:
                device = "cpu"
 
            reno_input = input_dict["reno_input"].to(device=device)
            self.decoder.to(device)
        
            # Regenerate offset key
            batch_indices = reno_input.coords[:, 0]
            counts = torch.bincount(batch_indices.long())
            offset = torch.cat([torch.tensor([0], device=device), counts.cumsum(dim=0)]).long()
            input_dict["offset"] = offset

        
            # 1a. Calculate Reconstruction Loss
            # This single line simulates reno/train.py to get the compression loss.
            loss_reconstruction = self.decoder(reno_input)


            # 1b. Generate Ground-Truth Occupancy Codes
            # This simulates the encoder creating the data that would normally be in a bitstream.
            with torch.no_grad():
                data_ls = []
                x_down = reno_input
                while True:
                    x_down = self.decoder.fog(x_down)
                    data_ls.append((x_down.coords.clone(), x_down.feats.clone()))
                    if x_down.coords.shape[0] < 64: break
                data_ls = data_ls[::-1]

            # 1c. Simulate Feature Decoding (Full Multi-scale Loop)
            # This adapts the logic from decompress.py. Instead of decoding from a bitstream,
            # we use the ground-truth occupancy codes from data_ls for a "perfect" decode.
            x_C, x_O = data_ls[0]
            x = SparseTensor(coords=x_C, feats=x_O) # Start with the coarsest scale

            for depth in range(len(data_ls) - 1):
                x_F = self.decoder.prior_embedding(x.F.int()).view(-1, self.decoder.channels)
                x_context = SparseTensor(coords=x.C, feats=x_F)
                x_context = self.decoder.prior_resnet(x_context)

                x_up_C, x_up_F = self.decoder.fcg(x_context.C, x.F.int(), x_context.F)
                x_up_F = self.decoder.target_embedding(x_up_F, x_up_C)
                x_up_context = SparseTensor(coords=x_up_C, feats=x_up_F)
                x_up_context = self.decoder.target_resnet(x_up_context)

                # Get the ground-truth occupancy for the next level
                _, gt_x_up_O = data_ls[depth + 1]

                # Update x for the next iteration of the loop
                x = SparseTensor(coords=x_up_context.C, feats=gt_x_up_O)

            # The final `x_up_context` contains the high-resolution decoded features
            decoded_features_sparse = x_up_context

            # ======================= STAGE 2: PTV3 SEGMENTATION =======================
            # 2a. Correctly Pad Features to Add a Synthetic Intensity Channel
            # We operate on the dense feature tensor (.F), not the sparse tensor object itself.
            reno_features = decoded_features_sparse.F
            decoded_coords = decoded_features_sparse.C

            # Use the Z-coordinate as a proxy for intensity
            # Note: torchsparse coords are (Batch, Z, Y, X), so index 1 is Z.
            z_channel = decoded_coords[:, 1:2].float()
            z_normalized = (z_channel - z_channel.min()) / (z_channel.max() - z_channel.min() + 1e-6)

            # Combine RENO features with the synthetic intensity
            # The final features passed to PTv3 have shape [N, 32 + 1]
            ptv3_feat = torch.cat([reno_features, z_normalized], dim=1)

            # 2b. Create a new Point object for the PTv3 backbone
            ptv3_input_dict = {
                "coord": decoded_coords,
                "feat": ptv3_feat,
                "grid_coord": decoded_coords[:, 1:],
                "offset": input_dict["offset"]
            }
            point = Point(ptv3_input_dict)

            # Run PTv3 backbone and segmentation head
            point = self.segmentation_backbone(point)

            seg_logits = self.seg_head(point.feat)

            # ======================= STAGE 3: LOSS =======================
            if self.training:
                # Labels per-voxel
                # Most common label for all points in a voxel is chosen
                num_voxels = seg_logits.shape[0]
                voxel_segment = torch.zeros(num_voxels, dtype=torch.long, device=seg_logits.device)
                # scatter_max is fast majority vote
                voxel_segment, _ = torch_scatter.scatter_max(
                    input_dict["segment"].to(seg_logits.device),
                    input_dict["inverse_map"].to(seg_logits.device),
                    out=voxel_segment
                )
                               
                
                loss_seg = self.criteria(seg_logits, voxel_segment)
                total_loss = loss_seg + self.recon_loss_weight * loss_reconstruction
                
                # Return individual losses for logging
                #return {"loss": total_loss, "loss_seg": loss_seg, "loss_recon": loss_reconstruction}
                
                # Only return loss for ptv3d 
                return {"loss": loss_seg}
            else: # For validation
                # Project the voxel predictions back to original point cloud space
                inverse_map = input_dict["inverse_map"].to(seg_logits.device)
                point_logits = seg_logits[inverse_map]
                target_labels = input_dict["segment"].to(seg_logits.device)
                
                loss_seg = self.criteria(point_logits, target_labels)
                #loss_seg = self.criteria(seg_logits, input_dict["segment"].squeeze(0))
                return {"loss": loss_seg, "seg_logits": point_logits}


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)

        if self.training:
            try:
                loss = self.criteria(seg_logits, input_dict["segment"])
            except:
                print(f"LOSS SHAPE ERROR: seg_logits_shape: {seg_logits.shape}, segment_shape: {input_dict['segment'].shape}")
                loss = torch.tensor(float('nan'))
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
