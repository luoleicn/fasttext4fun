cmake_minimum_required(VERSION 3.0)

set (PROJECT_NAME fasttext)

PROJECT(${PROJECT_NAME} CXX )
#PROJECT(${PROJECT_NAME} CUDA)


SET(CMAKE_C_COMPILER "g++")  
SET(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")  
set (CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11)

#debug
set(CUDA_NVCC_FLAGS -G;-g)
set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS}  -Wall -g2 -ggdb")


#find_package(CUDA REQUIRED)
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#SET(CUDA_HOST_COMPILER /usr/bin/g++)
#LIST(APPEND CUDA_NVCC_FLAGS "-use_fast_math -std=c++11 -ftz=true -prec-div=false -prec-sqrt=false")
#add_definitions(-DRS2_USE_CUDA)

#LIST(APPEND COMPILE_FLAGS " -fPIC")
#LIST(APPEND CUDA_NVCC_FLAGS " -fPIC")
#set(LDFLAGS "-fPIC")

message("project name : " ${PROJECT_NAME})
message("project dir : " ${PROJECT_SOURCE_DIR})
message("CMAKE_SOURCE_DIR dir : " $ENV{CMAKE_SOURCE_DIR})


set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# 需包含的头文件
INCLUDE_DIRECTORIES(
	${PROJECT_SOURCE_DIR}/include
	/usr/include/eigen3
)

# 需包含的库文件
link_directories(
    ${PROJECT_BINARY_DIR}/lib
    /usr/local/lib
    /usr/local/cuda/lib64
    /usr/lib
)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
#ADD_SUBDIRECTORY(python)

