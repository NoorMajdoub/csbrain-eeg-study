import re

path = "csbrain_optuna_search.py"
with open(path) as f:
    src = f.read()

orig = src

# 1. build_params signature + datasets_dir resolution
old = 'def build_params(dataset_key, arm, lr, clip_value):'
new = 'def build_params(dataset_key, arm, lr, clip_value, fraction="100"):'
assert src.count(old) == 1, "build_params signature not found or not unique"
src = src.replace(old, new)

old = '        datasets_dir=cfg["root"],'
new = '        datasets_dir=cfg["root"] if fraction == "100" else f\'{cfg["root"]}_{fraction}pct\','
assert src.count(old) == 1, "datasets_dir line not found or not unique"
src = src.replace(old, new)

# 2. objective signature + build_params call
old = 'def objective(trial, dataset_key, arm):'
new = 'def objective(trial, dataset_key, arm, fraction="100"):'
assert src.count(old) == 1, "objective signature not found or not unique"
src = src.replace(old, new)

old = '    params = build_params(dataset_key, arm, lr, clip_value)'
new = '    params = build_params(dataset_key, arm, lr, clip_value, fraction)'
assert src.count(old) == 1, "build_params call not found or not unique"
src = src.replace(old, new)

# 3. argparse --fraction
old = '    parser.add_argument("--n_trials", type=int, default=20)'
new = old + '\n    parser.add_argument("--fraction", default="100", choices=["100", "50", "25"])'
assert src.count(old) == 1, "--n_trials add_argument not found or not unique"
src = src.replace(old, new)

# 4. tag
old = '    tag = f"{cli_args.dataset}_{cli_args.arm}_100pct"'
new = '    tag = f"{cli_args.dataset}_{cli_args.arm}_{cli_args.fraction}pct"'
assert src.count(old) == 1, "tag line not found or not unique"
src = src.replace(old, new)

# 5. lambda passed to study.optimize
old = '            lambda trial: objective(trial, cli_args.dataset, cli_args.arm),'
new = '            lambda trial: objective(trial, cli_args.dataset, cli_args.arm, cli_args.fraction),'
assert src.count(old) == 1, "objective lambda not found or not unique"
src = src.replace(old, new)

# 6. results json "fraction" field
old = '                "fraction": "100",'
new = '                "fraction": cli_args.fraction,'
assert src.count(old) == 1, "fraction json field not found or not unique"
src = src.replace(old, new)

assert src != orig, "no changes applied"
with open(path, "w") as f:
    f.write(src)

print("Patch applied successfully.")
