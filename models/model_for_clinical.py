import torch
import torch.nn as nn
from .CSBrain import *

# Channel order verified against ds004504's actual channels.tsv on GitHub
# (OpenNeuroDatasets/ds004504), standard 19-channel 10-20 montage:
# https://github.com/OpenNeuroDatasets/ds004504
clinical_electrode_labels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                              'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                              'Fz', 'Cz', 'Pz']

clinical_brain_regions = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 4, 4, 4, 0, 1, 2]
# 0: Frontal, 1: Central, 2: Parietal, 3: Occipital, 4: Temporal

clinical_topology = {
    0: ['Fp1', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Fp2'],
    1: ['C3', 'Cz', 'C4'],
    2: ['P3', 'Pz', 'P4'],
    3: ['O1', 'O2'],
    4: ['T3', 'T5', 'T4', 'T6'],
}

clinical_region_groups = {}
for i, region in enumerate(clinical_brain_regions):
    clinical_region_groups.setdefault(region, []).append((i, clinical_electrode_labels[i]))

clinical_sorted_indices = []
for region in sorted(clinical_region_groups.keys()):
    electrodes = clinical_region_groups[region]
    sorted_electrodes = sorted(electrodes, key=lambda x: clinical_topology[region].index(x[1]))
    clinical_sorted_indices.extend([e[0] for e in sorted_electrodes])


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=clinical_brain_regions,
                sorted_indices=clinical_sorted_indices
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
            nn.Linear(19 * 10 * 200, 10 * 200),
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