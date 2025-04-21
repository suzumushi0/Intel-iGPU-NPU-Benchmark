# Intel iGPU and NPU Benchmark

2025/4/25 に開催された C++ MIX #14 の「Data Parallel C++ と OpenVINO で iGPU, NPU の計算速度とエネルギー効率を測ってみた」で使用したソースコードです．ご自由にお使いください．

https://cppmix.connpass.com/event/349703/

## ソースコードの内容

`DPC++/CPU_matrix.cpp` 
CPU による正方行列積，Intel C++ OpenMP によりスレッド化

`DPC++/GPU_matrix_OpenMP.cpp` 
GPU による正方行列積 (転置行列)，Intel DPC++ OpenMP による GPU 処理

`DPC++/GPU_matrix_SYCL.cpp` 
GPU による正方行列積，Intel DPC++ SYCL による GPU 処理

`OpenVINO/xPU_matrix_OpenVINO.cpp`
CPU, GPU, NPU による正方行列積 (転置行列)，Microsoft VC++ から OpenVINO C++ API を利用


## VisualStudio によるコンパイル方法

### Intel C++ OpenMP によるスレッド化

|C\++コンソールアプリを作成，Intel C\++ に切り替え，Release 版に以下を設定 ||
|:-----------|:------------|
| [構成プロパティ]>[General]>[C++ Language Standard]			| /std:c++latest |	
| [構成プロパティ]>[C/C++]>[General]>[Suppress Startup Banner]	| No |	
| [構成プロパティ]>[C/C++]>[Code Generation]>[Enable Enhanced Instruction Set] | /arch:AVX2 |	
| [構成プロパティ]>[C/C++]>[Code Generation]>[Floating Point Model] | /fp:fast |	
| [構成プロパティ]>[C/C++]>[Output Files]>[Assembler Output] | /FAs |	
| [構成プロパティ]>[C/C++]>[Diagnostics [Intel C++]]>[Optimization Diagnostics Level] | /Qopt-report:1 |	

| OpenMP によるスレッド化の設定 ||
|:-----------|:------------|
| [構成プロパティ]>[C/C++]>[Language [Intel C++]]>[OpenMP Support] | /Qiopenmp |	
| [構成プロパティ]>[C/C++]>[Language [Intel C++]]>[Enable OpenMP Offloading] | /Qopenmp-targets: spir64_x86_64 |
| [構成プロパティ]>[Linker]>[DPC++]>[Specify CPU target device for AOT compilation] | /Xs "-march=avx2" |

### Intel DPC++ OpenMP による GPU 処理

|DPC\++コンソールアプリを作成，Release 版に以下を設定 ||													
|:-----------|:------------|
| [構成プロパティ]>[DPC++]>[General]>[Suppress Startup Banner] | No |
| [構成プロパティ]>[DPC++]>[Diagnostics]>[Optimization Diagnostics Level] |	/Qopt-report:1 |
| [構成プロパティ]>[DPC++]>[Command Line]>[Additional Options] | /arch:avx2 /fp:fast を追加 |			
| [構成プロパティ]>[DPC++]>[Language]>[C++ Language Standard] | /std:c++latest |			
| [構成プロパティ]>[DPC++]>[Language]>[OpenMP Support] | /Qiopenmp |			
| **JIT コンパイル時** ||
| [構成プロパティ]>[DPC++]>[Language]>[Enable OpenMP Offloading] | /Qopenmp-targets: spir64_x86_64 |			
| **AOT コンパイル時** ||
| [構成プロパティ]>[DPC++]>[Language]>[Enable OpenMP Offloading] | /Qopenmp-targets: spir64_gen |			
| [構成プロパティ]>[Linker]>[General]>[Pass \<arg> to the device code linker for OpenMP offload | -v -device `GPU_device_name` |			

`GPU_device_name` は以下を参照

https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-1/ahead-of-time-compilation.html

### Intel DPC++ SYCL による GPU 処理

|DPC\++コンソールアプリを作成，Release 版に以下を設定 ||													
|:-----------|:------------|
| [構成プロパティ]>[DPC++]>[General]>[Suppress Startup Banner] | No |
| [構成プロパティ]>[DPC++]>[Diagnostics]>[Optimization Diagnostics Level] |	/Qopt-report:1 |
| [構成プロパティ]>[DPC++]>[Command Line]>[Additional Options] | /arch:avx2 /fp:fast を追加 |			
| [構成プロパティ]>[DPC++]>[Language]>[C++ Language Standard] | /std:c++latest |					
| **JIT コンパイル時** ||
| [構成プロパティ]>[DPC++]>[General]>[Specify SYCL offloading targets for AOT] | default |
| **AOT コンパイル時** ||
| [構成プロパティ]>[DPC++]>[General]>[Specify SYCL offloading targets for AOT] | /fsycl-targets=spire64_gen |
| [構成プロパティ]>[Linker]>[General]>[Pass \<arg> to the device code linker] | -v -device `GPU_device_name` |			

`GPU_device_name` は以下を参照

https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-1/ahead-of-time-compilation.html

### Microsoft VC++ から OpenVINO C++ API を利用

|C\++コンソールアプリを作成，Release 版に以下を設定 ||
|:-----------|:------------|
| [構成プロパティ]>[General]>[C++ Language Standard] | /std:c++20 |		
| [構成プロパティ]>[C/C++]>[Code Generation]>[Enable Enhanced Instruction Set] | /arch:AVX2 |		
| [構成プロパティ]>[C/C++]>[Code Generation]>[Floating Point Model] | /fp:fast |		
| [構成プロパティ]>[C/C++]>[General]>[Additional Include Directories] | `OpenVINO_Install_dir`\runtime\include |
| [構成プロパティ]>[Linker]>[General]>[Additional Library Directories] | `OpenVINO_Install_dir`\runtime\lib\intel64\Release		
| [構成プロパティ]>[Linker]>[Input]>[Additional Dependencies] | openvino.lib |

若しくは，setupvars.ps1 により環境変数を設定し，cmake を実行




