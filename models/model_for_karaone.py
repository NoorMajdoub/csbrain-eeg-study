import torch
import torch.nn as nn
from .CSBrain import *

# Channel order taken directly from REVE's own karaone.yaml task config
# (electrodes: [...]), which is what REVE relied on for its reve-positions
# lookup -- reused here since REVE reads the same raw KaraOne source data.
# 60 channels, standard 10-10 system.
karaone_electrode_labels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
                             'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                             'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                             'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                             'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                             'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'OZ', 'O2', 'O1']

# Regions assigned by anteroposterior band (Fp < AF < F < FC/FT < C/T < CP/TP
# < P < PO < O), derived programmatically from each label's prefix -- 9
# bands present in this 10-10 montage.
karaone_brain_regions = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8]

# Within each band, ordered left-to-right using the standard 10-10 parity
# convention (odd suffix = left hemisphere, higher number = more lateral;
# even suffix = right hemisphere; no suffix/'Z' = midline).
karaone_sorted_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 59, 57, 58]


class Model(nn.Module):
    """
    KaraOne imagined-speech binary classification, 60 channels x 5 patches
    (see karaone_dataset.py -- resampled to a fixed 5s @ 200Hz).
    """
    def __init__(self, param):
        super(Model, self).__init__()

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=karaone_brain_regions,
                sorted_indices=karaone_sorted_indices
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
            nn.Linear(60 * 5 * 200, 5 * 200),
            nn.ELU(),
            nn.Linear(5 * 200, 200),
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