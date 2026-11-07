"""Idempotent patch script for CSBrain's finetune_main.py -- adds the
import lines and elif dispatch blocks for the 6 new datasets (KaraOne,
Yang2025, Dream, JapanEEG, Clinical, ThoughtViz). Safe to run more than
once: it checks for its own marker before inserting anything.
Run from inside the CSBrain repo root: python3 patch_finetune_main.py
"""
import re

PATH = "finetune_main.py"

IMPORT_ANCHOR = "from models import model_for_siena, model_for_hmc,model_for_stress,model_for_mumtaz"
NEW_IMPORTS = (
    "from datasets import karaone_dataset, yang2025_dataset, dream_dataset, "
    "japaneeg_dataset, clinical_dataset, thoughtviz_dataset\n"
    "from models import model_for_karaone, model_for_yang2025, model_for_dream, "
    "model_for_japaneeg, model_for_clinical, model_for_thoughtviz"
)

ELIF_ANCHOR = '    print("model:", params.model, "seed:", params.seed, "lr:", params.lr,"weight_decay:",params.weight_decay, "dropout:", params.dropout, "foundation_dir:", params.foundation_dir)'
NEW_ELIFS = """    elif params.downstream_dataset == 'KaraOne':
        load_dataset = karaone_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_karaone.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'Yang2025':
        load_dataset = yang2025_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_yang2025.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'Dream':
        load_dataset = dream_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_dream.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'JapanEEG':
        load_dataset = japaneeg_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_japaneeg.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'Clinical':
        load_dataset = clinical_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_clinical.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ThoughtViz':
        load_dataset = thoughtviz_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_thoughtviz.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
"""

with open(PATH) as f:
    src = f.read()

if "model_for_karaone" in src:
    print("Already patched (found model_for_karaone import) -- doing nothing.")
else:
    assert IMPORT_ANCHOR in src, "import anchor line not found -- file may have changed, patch manually"
    assert ELIF_ANCHOR in src, "elif anchor line not found -- file may have changed, patch manually"

    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "\n" + NEW_IMPORTS, 1)
    src = src.replace(ELIF_ANCHOR, NEW_ELIFS + ELIF_ANCHOR, 1)

    with open(PATH, "w") as f:
        f.write(src)
    print("Patched finetune_main.py successfully.")