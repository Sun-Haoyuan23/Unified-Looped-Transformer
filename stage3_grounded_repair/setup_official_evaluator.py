"""Create an isolated official GenEval runtime; never edit another project venv."""
import json
import subprocess
import sys
import time
from pathlib import Path

work = Path(sys.argv[1]).resolve()
target = work / 'stage3_grounded_repair/evaluator_runtime_v1'
target.mkdir(exist_ok=True)
status = target / 'setup_status.json'
commands = [
    [sys.executable, '-m', 'venv', str(target / 'venv')],
    [str(target/'venv/bin/python'), '-m', 'pip', 'install', '--no-cache-dir',
     'torch==1.13.1+cu117', 'torchvision==0.14.1+cu117',
     '--extra-index-url', 'https://download.pytorch.org/whl/cu117'],
    [str(target/'venv/bin/python'), '-m', 'pip', 'install', '--no-cache-dir',
     'mmcv-full==1.7.1', '--only-binary', 'mmcv-full',
     '-f', 'https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html'],
    [str(target/'venv/bin/python'), '-m', 'pip', 'install', '--no-cache-dir',
     'mmdet==2.28.2', 'numpy==1.23.5', 'pandas==1.5.3', 'Pillow==9.4.0',
     'yapf==0.33.0', 'opencv-python==4.6.0.66', 'matplotlib==3.7.1',
     'scipy==1.9.1', 'scikit-image==0.20.0', 'open-clip-torch==2.20.0',
     'clip-benchmark==1.4.0', 'timm==0.9.2', 'transformers==4.36.1',
     'huggingface-hub==0.19.4'],
]
def update(phase, **kw):
    status.write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2))
try:
    (target/'setup_commands.json').write_text(json.dumps(commands,indent=2))
    for index, command in enumerate(commands):
        update('installing',step=index,command=command)
        with (target/f'step{index}.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    with (target/'pip_freeze.txt').open('w') as output:
        subprocess.run([str(target/'venv/bin/python'),'-m','pip','freeze'],check=True,stdout=output)
    update('installed_not_yet_semantically_verified',steps=len(commands))
except Exception as error:
    update('failed',error=str(error));raise
