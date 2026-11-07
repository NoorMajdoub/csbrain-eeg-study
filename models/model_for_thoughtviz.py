import torch
import torch.nn as nn
from .CSBrain import *

# Emotiv EPOC's 14-channel layout is fixed by the hardware itself (every
# Emotiv EPOC unit streams channels in this exact order) -- this is not
# dataset-specific to ThoughtViz, it's the device's standard channel order:
# AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
thoughtviz_electrode_labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                                'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

thoughtviz_brain_regions = [0, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0]
# 0: Frontal, 1: Fronto-central, 2: Temporal, 3: Parietal, 4: Occipital

thoughtviz_topology = {
    0: ['AF3', 'F7', 'F3', 'F4', 'F8', 'AF4'],
    1: ['FC5', 'FC6'],
    2: ['T7', 'T8'],
    3: ['P7', 'P8'],
    4: ['O1', 'O2'],
}

thoughtviz_region_groups = {}
for i, region in enumerate(thoughtviz_brain_regions):
    thoughtviz_region_groups.setdefault(region, []).append((i, thoughtviz_electrode_labels[i]))

thoughtviz_sorted_indices = []
for region in sorted(thoughtviz_region_groups.keys()):
    electrodes = thoughtviz_region_groups[region]
    sorted_electrodes = sorted(electrodes, key=lambda x: thoughtviz_topology[region].index(x[1]))
    thoughtviz_sorted_indices.extend([e[0] for e in sorted_electrodes])


class Model(nn.Module):
    """
    ThoughtViz: 14 channels x 1 patch (only 25% of that single patch is
    real signal -- see thoughtviz_dataset.py). 10-class multiclass output.
    """
    def __init__(self, param):
        super(Model, self).__init__()

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=thoughtviz_brain_regions,
                sorted_indices=thoughtviz_sorted_indices
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
            nn.Linear(14 * 1 * 200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * 200)
        out = self.classifier(feats)
        return out