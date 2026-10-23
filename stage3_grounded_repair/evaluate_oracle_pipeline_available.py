"""Resource-only override: same frozen evaluator, four currently free GPUs."""
import evaluate_oracle_pipeline as pipeline
from launch_grounding_diagnostic import GPUS
pipeline.GPUS=[GPUS[g] for g in (0,4,6,7)]
if __name__=='__main__':pipeline.main()
