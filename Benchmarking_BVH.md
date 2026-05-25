gracy(s205465) $ mkdir -p $HOME/.tmp-ncu
TMPDIR=$HOME/.tmp-ncu ncu ./build/raytrace
Number of scene objects: 488
Scene: cover
Samples per pixel : 10
==PROF== Connected to process 152070 (/zhome/98/f/155814/cuda/CUDA_Project/build/raytrace)
==PROF== Profiling "init_rng_kernel" - 0: 0%....50%....100% - 10 passes
==PROF== Profiling "render_kernel" - 1: 0%....50%....100% - 10 passes
CUDA render time: 0.726997 s
==PROF== Disconnected from process 152070
[152070] raytrace@127.0.0.1
  init_rng_kernel(RngState *, int, int, unsigned int) (75, 43, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 9.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         2.59
    SM Frequency                    Ghz         1.50
    Elapsed Cycles                cycle        7,145
    Memory Throughput                 %        17.75
    DRAM Throughput                   %         0.01
    Duration                         us         4.74
    L1/TEX Cache Throughput           %        22.22
    L2 Cache Throughput               %        24.39
    SM Active Cycles              cycle     3,460.93
    Compute (SM) Throughput           %        28.52
    ----------------------- ----------- ------------

    OPT   This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak           
          performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak           
          typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential       
          reasons.                                                                                                      

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  3,225
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM             132
    Stack Size                                                 1,024
    Threads                                   thread         825,600
    # TPCs                                                        66
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                3.05
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 25%                                                                                             
          A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the    
          target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical       
          occupancy of the kernel. This kernel launch results in 3 full waves and a partial wave of 58 thread blocks.   
          Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for  
          up to 25.0% of the total runtime of this kernel. Try launching a grid with no partial wave. The overall       
          impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware   
          Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for     
          more details on launch configurations.                                                                        

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit Barriers                  block           32
    Block Limit SM                        block           32
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block            8
    Theoretical Active Warps per SM        warp           64
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        50.14
    Achieved Active Warps Per SM           warp        32.09
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 49.86%                                                                                    
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (50.1%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle         1.83
    Total DRAM Elapsed Cycles        cycle      589,824
    Average L1 Active Cycles         cycle     3,460.93
    Total L1 Elapsed Cycles          cycle      939,726
    Average L2 Active Cycles         cycle     4,112.84
    Total L2 Elapsed Cycles          cycle      761,568
    Average SM Active Cycles         cycle     3,460.93
    Total SM Elapsed Cycles          cycle      939,726
    Average SMSP Active Cycles       cycle     3,555.19
    Total SMSP Elapsed Cycles        cycle    3,758,904
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 5.761%                                                                                          
          One or more L2 Slices have a much higher number of active cycles than the average number of active cycles.    
          Maximum instance value is 11.11% above the average, while the minimum instance value is 5.81% below the       
          average.                                                                                                      

  render_kernel(uchar3 *, GpuCamera, GpuScene, RngState *) (75, 43, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 9.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          2.62
    SM Frequency                    Ghz          1.53
    Elapsed Cycles                cycle    17,508,078
    Memory Throughput                 %         35.10
    DRAM Throughput                   %          0.01
    Duration                         ms         11.44
    L1/TEX Cache Throughput           %         37.07
    L2 Cache Throughput               %          0.11
    SM Active Cycles              cycle 16,578,748.80
    Compute (SM) Throughput           %         64.86
    ----------------------- ----------- -------------

    OPT   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the   
          compute pipelines are spending their time doing. Also, consider whether any computation is redundant and      
          could be reduced or moved to look-up tables.                                                                  

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  3,225
    Registers Per Thread             register/thread              67
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM             132
    Stack Size                                                 1,024
    Threads                                   thread         825,600
    # TPCs                                                        66
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                8.14
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit Barriers                  block           32
    Block Limit SM                        block           32
    Block Limit Registers                 block            3
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block            8
    Theoretical Active Warps per SM        warp           24
    Theoretical Occupancy                     %        37.50
    Achieved Occupancy                        %        33.23
    Achieved Active Warps Per SM           warp        21.27
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 62.5%                                                                                     
          The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 16. This kernel's theoretical occupancy (37.5%) is limited by the number of required      
          registers.                                                                                                    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle      2,145.17
    Total DRAM Elapsed Cycles        cycle 1,438,555,648
    Average L1 Active Cycles         cycle 16,578,748.80
    Total L1 Elapsed Cycles          cycle 2,311,026,956
    Average L2 Active Cycles         cycle    395,702.07
    Total L2 Elapsed Cycles          cycle 1,874,088,768
    Average SM Active Cycles         cycle 16,578,748.80
    Total SM Elapsed Cycles          cycle 2,311,026,956
    Average SMSP Active Cycles       cycle 16,375,907.10
    Total SMSP Elapsed Cycles        cycle 9,244,107,824
    -------------------------- ----------- -------------

~/cuda/CUDA_Project
gracy(s205465) $ 