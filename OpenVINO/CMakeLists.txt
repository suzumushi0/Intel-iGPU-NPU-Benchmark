
cmake_minimum_required (VERSION 3.14.0)

set_property (GLOBAL PROPERTY USE_FOLDERS ON)
source_group ("Predefined" ALL_BUILD ZERO_CHECK)

project (OpenVINO_benchmark)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "/arch:AVX2 /fp:fast")

find_package (OpenVINO REQUIRED)

add_executable (${PROJECT_NAME} xPU_matrix_OpenVINO.cpp)

target_link_libraries (${PROJECT_NAME} PRIVATE openvino::runtime)

