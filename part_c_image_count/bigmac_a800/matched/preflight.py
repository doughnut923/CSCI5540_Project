import hashlib
import json
import os
from pathlib import Path
import sys

import torch

root = Path(os.environ['PARTC_RUN_DIR'])
primary = Path(os.environ['PARTC_PRIMARY'])
assert torch.cuda.device_count() == 2, f'Expected two platform-assigned GPUs, got {torch.cuda.device_count()}'
gpus = []
for rank in range(2):
    properties = torch.cuda.get_device_properties(rank)
    assert 'A800' in properties.name, properties
    gpus.append(dict(rank=rank, name=properties.name, uuid=str(properties.uuid), memory=properties.total_memory))
assert gpus[0]['uuid'] != gpus[1]['uuid']
verification = json.loads((primary/'data/verification.json').read_text())
for condition in verification['cases'].values():
    for sample in condition['cases'].values():
        path = primary/'data'/sample['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample['sha256']
for gpu in gpus:
    del gpu['uuid']
allocation = dict(distinct_devices_verified=True, gpus=gpus, python=sys.version, torch=torch.__version__, cuda=torch.version.cuda)
(root/'results/allocation.json').write_text(json.dumps(allocation, indent=2)+'\n')
print(allocation, flush=True)
