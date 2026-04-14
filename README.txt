Usage guide:

If you're on the HPC cluster (gracy), first run:
source load_modules.sh

This loads:
- nvhpc/25.9
- ffmpeg/7.1.1

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