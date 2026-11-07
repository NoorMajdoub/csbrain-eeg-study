import torch
import torch.nn as nn
from .CSBrain import *

# Region grouping + sorted_indices derived geometrically from JapanEEG's
# APPROXIMATE electrode 3D coordinates (japaneeg_positions_APPROXIMATE.npy,
# 128 channels) via k-means clustering (13 regions, ~10 electrodes/region)
# on (x, y, z), region IDs assigned anterior->posterior (ascending region-ID
# sweeps front-to-back), same method as DREAM -- see
# montage_from_positions.py. NOTE: positions are labeled APPROXIMATE by
# whoever supplied them, so this montage is a best-effort spatial grouping,
# not an exact one -- fine for CSBrain's region-local attention (it doesn't
# need anatomically perfect boundaries), but flag this if results look off.
japaneeg_brain_regions = [7, 7, 7, 7, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 9, 9, 9, 7, 7, 9, 9, 9, 9, 12, 12, 12, 12, 9, 9, 9, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 2, 7, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 7, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 11, 11]

japaneeg_sorted_indices = [80, 81, 92, 91, 82, 90, 83, 93, 94, 89, 95, 79, 78, 77, 71, 72, 73, 70, 69, 68, 76, 84, 75, 67, 66, 74, 63, 65, 88, 85, 98, 87, 86, 108, 97, 99, 102, 101, 100, 106, 103, 104, 105, 107, 58, 59, 60, 55, 56, 57, 47, 46, 45, 61, 62, 51, 52, 53, 54, 49, 50, 48, 96, 64, 113, 109, 0, 110, 32, 1, 111, 33, 112, 2, 3, 118, 117, 116, 115, 114, 123, 121, 120, 119, 122, 124, 125, 34, 44, 43, 42, 35, 31, 36, 30, 37, 29, 5, 6, 4, 18, 7, 17, 19, 16, 20, 127, 126, 11, 10, 8, 9, 12, 13, 15, 14, 41, 40, 39, 38, 25, 24, 26, 28, 27, 23, 21, 22]


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=japaneeg_brain_regions,
                sorted_indices=japaneeg_sorted_indices
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
            nn.Linear(128 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * 200)
        out = self.classifier(feats)
        out = out.contiguous().view(bz)
        return out