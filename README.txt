Usage guide:

If you're on the HPC cluster (gracy), first run:
source load_modules.sh

This loads:
- nvhpc/25.9

Then build using:

make (all)
make cpu (cpu only)
make cuda (cuda only)

Then run using the versatile make target with value-based argument parsing:

make run [args...]

Arguments (order doesn't matter):
- scene:  simple or cover (default: cover)
- mode:   cpu, omp, cuda, or all (default: cuda)
- samples: integer >= 1 for quality (default: 10)

Examples:
make run simple cuda 100           # Simple scene, CUDA, 100 samples
make run 50 cover omp              # Coverpage scene, OMP, 50 samples
make run cuda 200                  # Coverpage scene (default), CUDA, 200 samples
make run simple                    # Simple scene, CUDA, 10 samples (defaults)
make run                           # Coverpage scene, CUDA, 10 samples (all defaults)


-------------------
For profiling, you might need to remove a stale lock file before running Nsight Compute:

rm -f /tmp/nsight-compute-lock

Then run ncu to profile

ncu ./build/raytrace

If the lockfile is owned by someone else, you can't delete it. In this case, create a temp directory 
and use that to run NCU.

source load_modules.sh
mkdir -p $HOME/.tmp-ncu
TMPDIR=$HOME/.tmp-ncu ncu ./build/raytrace cover 500

To get an output file that can be viewed with Nsight Compute GUI:

mkdir -p $HOME/.tmp-ncu
TMPDIR=$HOME/.tmp-ncu ncu -o profile_test --page=details ./build/raytrace cover 500