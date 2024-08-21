import cmake

cmake -G "Visual Studio 17 2022" -A x64 `
  -DOpenBLAS_DIR="C:\Users\YH006_new\Downloads\OpenBLAS-0.3.28-x64-64\lib\cmake\openblas" `
  -DCMAKE_BUILD_TYPE = Release..