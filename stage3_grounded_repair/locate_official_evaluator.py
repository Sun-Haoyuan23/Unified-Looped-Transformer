"""Read-only discovery of candidate GenEval environments; not compatibility proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);args=p.parse_args()
    env=os.environ.copy();env.update(PYTHONPATH=str(args.work/'deps'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    probe="""from PIL import Image
import torch,mmcv,mmdet,json
from pathlib import Path
from mmcv.ops import roi_align
from mmdet.apis import init_detector,inference_detector
base=Path(mmdet.__file__).parent
name='mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
paths=[base/'.mim/configs/mask2former'/name,base/'configs/mask2former'/name]
print(json.dumps({'torch':torch.__version__,'mmcv':mmcv.__version__,'mmdet':mmdet.__version__,
 'configs':[str(q) for q in paths if q.is_file()]}))
"""
    candidates=[Path('/home/shy_23/venvs')/e/'bin/python' for e in
                ('geneval2_official','corebench_official','compbench_full_official','show_o2')]
    output=[]
    for python in candidates:
        if not python.is_file():
            output.append({'python':str(python),'exists':False});continue
        try:
            q=subprocess.run([str(python),'-c',probe],capture_output=True,text=True,env=env,timeout=90)
            output.append({'python':str(python),'returncode':q.returncode,'stdout':q.stdout.strip(),'stderr':q.stderr[-1500:]})
        except subprocess.TimeoutExpired:
            output.append({'python':str(python),'error':'dependency probe timed out'})
    print(json.dumps({'candidates':output,'note':'Read-only import/config discovery; full144 frozen semantic anchors still required'},indent=2))


if __name__=='__main__':main()
