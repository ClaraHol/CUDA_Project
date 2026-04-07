Usage guide:

If you're on the HPC cluster (gracy), first run:
module load nvhpc/25.9

Then build using:

make (all)
make cpu (cpu only)
make cuda (cuda only)

Then run using:
make run-all (or run-cpu, run-omp, run-cuda)