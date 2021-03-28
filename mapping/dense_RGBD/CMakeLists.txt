cmake_minimum_required(VERSION 2.8)
project(pointcloud_mapping)

set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# opencv 
set(OpenCV_DIR "/usr/local/opencv4/share/opencv4")
find_package(OpenCV 4.5 REQUIRED)
include_directories("/usr/local/opencv4/include")

# eigen 
include_directories("/usr/include/eigen3/")

# pcl 
set(PCL_DIR "/usr/share/pcl-1.8")
find_package(PCL 1.8 REQUIRED)
include_directories("/usr/include/pcl-1.8")
add_definitions(${PCL_DEFINITIONS})

# octomap
find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

add_executable(pointcloud_mapping pointcloud_mapping.cpp)
target_link_libraries(pointcloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(surfel_mapping surfel_mapping.cpp)
target_link_libraries(surfel_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(octomap_mapping octomap_mapping.cpp)
target_link_libraries(octomap_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})
