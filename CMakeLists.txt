cmake_minimum_required(VERSION 2.8)
project(myslam)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH  ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories("/usr/include/eigen3")

set(OpenCV_DIR "/usr/local/opencv-3.4.12/share/OpenCV")
find_package(OpenCV 3.4.12 REQUIRED)
include_directories("usr/local/opencv-3.4.12/include")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS 
        ${OpenCV_LIBS} 
        ${Sophus_LIBRARIES} 
        ${Pangolin_LIBRARIES} GL GLU GLEW glut 
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension 
        ${GTEST_BOTH_LIBRARIES} 
        ${GLOG_LIBRARIES} 
        pthread 
        ${CSPARSE_LIBRARY})
        
enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)
