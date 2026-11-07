import torch
import torch.nn as nn
from .CSBrain import *

# Channel order taken directly from REVE's own yang2025.yaml task config
# (electrodes: [...] under data_loader.dataset), which is what REVE relied
# on for its reve-positions lookup -- reused here since REVE reads the same
# raw WBCIC-SHU source data. 59 real EEG channels, 10-05 montage.
yang2025_electrode_labels = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2',
                              'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3',
                              'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4',
                              'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
                              'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz',
                              'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2']

# Regions assigned by anteroposterior band (Fp < AF < F < FC/FT < C/T < CP/TP
# < P < PO < O), derived programmatically from each label's prefix -- 9
# bands present in this 10-05 montage.
yang2025_brain_regions = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8]

# Within each band, ordered left-to-right using the standard 10-10/10-05
# parity convention (odd suffix = left, higher number = more lateral; even
# suffix = right; no suffix/'z' = midline).
yang2025_sorted_indices = [1, 0, 2, 5, 3, 4, 6, 14, 12, 10, 8, 7, 9, 11, 13, 15, 23, 21, 19, 17, 16, 18, 20, 22, 24, 32, 30, 28, 26, 25, 27, 29, 31, 33, 40, 38, 36, 34, 35, 37, 39, 41, 47, 45, 43, 42, 44, 46, 48, 54, 52, 50, 49, 51, 53, 55, 57, 56, 58]


class Model(nn.Module):
    """
    Yang2025 / WBCIC-SHU binary (left-hand vs right-hand) motor imagery.
    59 channels x 6 patches (see yang2025_dataset.py -- resampled to 200Hz,
    6s trials = 6 whole 200-sample patches).
    """
    def __init__(self, param):
        super(Model, self).__init__()

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=yang2025_brain_regions,
                sorted_indices=yang2025_sorted_indices
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
            nn.Linear(59 * 6 * 200, 6 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(6 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)  # (bz, 59, 6, 200)
        feats = feats.contiguous().view(bz, -1)
        out = self.classifier(feats)
        out = out.contiguous().view(bz)
        return out