cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(resnet)
set(CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)

add_executable(resnet resnet.cpp)
target_compile_features(resnet PUBLIC cxx_range_for)
target_link_libraries(resnet ${TORCH_LIBRARIES})

