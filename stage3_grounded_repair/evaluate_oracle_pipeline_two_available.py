"""Only GPU resource pool differs; original official/scoring body unchanged."""
import evaluate_oracle_pipeline as pipeline
from launch_grounding_diagnostic import GPUS
import threading

# The original body needs four job slots (official/Pick/HPS/SigLIP).
# Repeating UUIDs preserves all metrics; one lock per UUID prevents concurrent
# scoring models from competing for the same device's memory.
pipeline.GPUS=[GPUS[g] for g in (6,7,6,7)]
OriginalExecutor=pipeline.concurrent.futures.ThreadPoolExecutor

class ResourceExecutor:
    def __init__(self,max_workers=4):
        self.pool=OriginalExecutor(max_workers=max_workers)
        self.locks={gpu:threading.Lock() for gpu in pipeline.GPUS}
    def __enter__(self):
        self.pool.__enter__();return self
    def __exit__(self,*args):return self.pool.__exit__(*args)
    def map(self,fn,jobs):
        def guarded(job):
            with self.locks[job[1]]:return fn(job)
        return self.pool.map(guarded,jobs)

pipeline.concurrent.futures.ThreadPoolExecutor=ResourceExecutor
if __name__=='__main__':pipeline.main()
