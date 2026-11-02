import json, sys
sys.path.insert(0, '.')
from ppt.loaders import dev_manifest
from ppt.features import build_cache
info = build_cache(dev_manifest('data/dev_raw.npz'), 'data/dev_cache120.npz', cap=120)
json.dump(info, open('data/dev_cache120_info.json', 'w'), indent=1); print(info)
