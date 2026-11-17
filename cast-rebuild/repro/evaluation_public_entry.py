"""Memory-only observer around the unchanged STEP58 public entry function."""
import argparse
import resource
from pathlib import Path
from repro.evaluation_common import write

def main():
    from repro.step58_entry import run
    import torch
    p=argparse.ArgumentParser()
    for name in ('plan-dir','sample-id','mode','private-model','patch-manifest','proposal-cache','output-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output_dir)
    torch.cuda.reset_peak_memory_stats()
    try:
        return run(Path(__file__).resolve().parents[1],Path(a.plan_dir),a.sample_id,a.mode,
            Path(a.private_model),Path(a.patch_manifest),Path(a.proposal_cache),out)
    finally:
        write(out/'memory_profile.json',{'status':'Measured_in_original_public_subprocess',
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'host_memory_scope':'public child process peak RSS; not summed Slurm/cgroup RAM',
            'model_generation_function_changed':False})

if __name__=='__main__':main()
