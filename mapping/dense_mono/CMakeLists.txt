project(dense_monocular)

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

include_directories("/usr/include/eigen3")

set(OpenCV_DIR "/usr/local/opencv4/share/opencv4")
find_package(OpenCV 4.5 REQUIRED)
include_directories("/usr/local/opencv4/include")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(dense_mapping dense_mapping.cpp)
target_link_libraries(dense_mapping ${OpenCV_LIBS} Sophus::Sophus)
