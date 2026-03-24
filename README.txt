Usage guide:

If you're on the HPC cluster, first run:
module load nvhpc/25.9

Then build using:

make (all)
make cpu (cpu only)
make cuda (cuda only)

Then run using:
./build/raytrace_cuda --mode cuda (or --mode cpu or --mode all)