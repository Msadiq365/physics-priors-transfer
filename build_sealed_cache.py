import json, sys
sys.path.insert(0, '.')
from ppt.loaders import sealed_manifest
from ppt.features import build_cache
info = build_cache(sealed_manifest('data/sealed_export.npz'), 'data/sealed_cache.npz')
json.dump(info, open('data/sealed_cache_info.json', 'w'), indent=1)
print({k: (v if k != 'skipped' else len(v)) for k, v in info.items()})
