import torch
import torch.nn as nn
from .CSBrain import *

# Region grouping + sorted_indices derived geometrically from DREAM's real
# electrode 3D coordinates (dream_positions.npy, EGI HydroCelGSN256v10 net,
# 257 channels) via k-means clustering (26 regions, ~10 electrodes/region)
# on (x, y, z), with region IDs assigned anterior->posterior so ascending
# region-ID order sweeps front-to-back (mirrors the manual left/midline/
# right topology dicts used for CHB-MIT/ISRUC, just derived from real
# coordinates instead of hand-picked since DREAM has no standard 10-20
# names). See montage_from_positions.py for the derivation script;
# regenerate these two lists together if the positions file changes.
dream_brain_regions = [7, 7, 2, 2, 2, 2, 6, 6, 6, 7, 2, 2, 2, 2, 6, 6, 6, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 5, 5, 4, 4, 4, 5, 5, 5, 6, 14, 14, 4, 4, 4, 5, 5, 12, 12, 14, 4, 4, 4, 12, 12, 12, 14, 4, 13, 13, 12, 12, 12, 9, 13, 13, 13, 12, 12, 9, 13, 13, 19, 19, 19, 14, 14, 14, 9, 13, 13, 22, 19, 19, 19, 14, 14, 9, 18, 18, 18, 22, 22, 22, 19, 19, 23, 23, 18, 18, 18, 18, 22, 22, 22, 23, 23, 18, 18, 18, 22, 22, 25, 25, 23, 23, 18, 18, 24, 24, 25, 25, 25, 23, 23, 23, 16, 16, 16, 18, 24, 24, 24, 25, 25, 25, 21, 21, 16, 16, 16, 24, 24, 24, 24, 25, 25, 21, 21, 21, 16, 16, 24, 24, 20, 20, 21, 21, 21, 21, 16, 24, 24, 20, 20, 20, 15, 15, 15, 15, 17, 17, 20, 20, 20, 15, 15, 15, 11, 11, 10, 10, 10, 17, 17, 17, 20, 15, 15, 15, 11, 11, 11, 10, 10, 17, 17, 17, 15, 11, 11, 11, 10, 10, 17, 17, 7, 7, 7, 11, 2, 2, 8, 8, 8, 7, 7, 7, 7, 2, 2, 7, 7, 8, 8, 8, 1, 1, 8, 8, 1, 1, 1, 8, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 3, 9, 9, 9, 9, 14]

dream_sorted_indices = [30, 31, 24, 25, 32, 18, 26, 19, 17, 20, 33, 27, 237, 238, 233, 239, 234, 229, 235, 230, 11, 12, 10, 13, 3, 2, 4, 223, 5, 222, 214, 213, 240, 241, 243, 242, 244, 247, 245, 251, 248, 246, 36, 37, 45, 38, 46, 53, 47, 54, 60, 55, 34, 28, 35, 39, 29, 40, 48, 41, 49, 21, 14, 22, 15, 6, 23, 7, 16, 42, 8, 9, 1, 0, 225, 221, 220, 224, 219, 211, 218, 210, 209, 236, 231, 226, 232, 227, 217, 228, 216, 215, 249, 252, 253, 250, 254, 66, 72, 255, 81, 90, 206, 205, 197, 196, 185, 184, 183, 212, 204, 203, 195, 202, 194, 193, 182, 181, 56, 50, 57, 51, 63, 58, 64, 70, 65, 71, 61, 67, 62, 68, 69, 73, 74, 82, 83, 43, 256, 44, 52, 59, 80, 79, 78, 89, 88, 201, 192, 180, 191, 172, 179, 190, 171, 178, 170, 169, 131, 143, 154, 163, 130, 142, 153, 129, 141, 208, 207, 200, 199, 198, 188, 187, 186, 174, 173, 91, 101, 92, 102, 93, 110, 103, 104, 111, 119, 112, 120, 132, 75, 77, 76, 87, 85, 86, 98, 97, 189, 177, 176, 168, 175, 167, 166, 158, 157, 162, 161, 152, 160, 140, 151, 159, 150, 139, 94, 84, 95, 96, 105, 113, 106, 107, 114, 99, 128, 100, 109, 127, 108, 118, 117, 126, 121, 133, 165, 122, 144, 164, 134, 156, 145, 155, 146, 135, 147, 115, 149, 123, 148, 116, 138, 125, 136, 124, 137]


class Model(nn.Module):
    """
    DREAM only has ~120 training trials (see CBraMod's model_for_dream.py
    for the original overfitting fix). Same fix carried over here: pool
    over both the channel dimension AND the time/patch dimension down to a
    single 200-dim vector per trial before a small MLP head, rather than
    flattening 257 channels x patches into a huge first Linear layer.
    """
    def __init__(self, param):
        super(Model, self).__init__()

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=dream_brain_regions,
                sorted_indices=dream_sorted_indices
            )
        else:
            return 0

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

            model_state_dict = self.backbone.state_dict()
            matching_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
            model_state_dict.update(matching_dict)
            self.backbone.load_state_dict(model_state_dict)

        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(200, 64),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)  # (bz, 257, patch_num, 200)
        feats = feats.mean(dim=2)  # average over time/patches -> (bz, 257, 200)
        feats = feats.mean(dim=1)  # average over channels -> (bz, 200)
        out = self.classifier(feats)
        out = out.contiguous().view(bz)
        return out