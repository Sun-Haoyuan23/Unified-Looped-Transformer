"""CPU-only portable export of v59 singular values; no model or inference."""
import json
from pathlib import Path
import torch
from candidate_provenance import file_hash
from runtime import write_json

ROOT=Path('/home/shy_23/iterative_visual_stage1_20261003/stage3_grounded_repair')
OUT=ROOT/'preservation_rank_audit_v59'


def main():
    if json.loads((OUT/'pipeline_status.json').read_text())['phase']!='complete':
        raise ValueError('Require complete frozen v59 audit')
    geometry=json.loads((OUT/'geometry_results.json').read_text())
    values={}
    for mode,row in geometry.items():
        path=OUT/(mode+'_spectra.pt')
        if file_hash(path)!=row['artifact_sha256']:
            raise ValueError('Original spectrum bytes changed')
        artifact=torch.load(path,map_location='cpu',weights_only=False)
        values[mode]={key:value.tolist() for key,value in artifact.items()}
    write_json(OUT/'spectra_portable.json',values)
    source=Path(__file__)
    (OUT/'source'/source.name).write_bytes(source.read_bytes())
    write_json(OUT/'portable_export_config.json',dict(source_sha256=file_hash(source),
        output_sha256=file_hash(OUT/'spectra_portable.json'),operation='Read unchanged CPU tensor spectra -> JSON',
        optimizer_updates=0,analytical_fits=0,new_scoring_images=0))


if __name__=='__main__':main()
