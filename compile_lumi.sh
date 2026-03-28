module load rocm
module load cray-python

rm -rf build
srun --gres=gpu:1 -A project_465002776 -p standard-g cmake -B build -DCMAKE_HIP_ARCHITECTURES=gfx90a
srun --gres=gpu:1 -A project_465002776 -p standard-g cmake --build build -j 8

