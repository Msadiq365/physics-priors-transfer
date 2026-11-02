import json, sys
sys.path.insert(0, '.')
from ppt.loaders import dev_manifest
from ppt.features import build_cache
man = dev_manifest('data/dev_raw.npz')
info = build_cache(man, 'data/dev_cache.npz')
json.dump(info, open('data/dev_cache_info.json', 'w'), indent=1)
print({k: v for k, v in info.items()})
