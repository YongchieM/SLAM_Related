project(orb)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(OpenCV_DIR "/usr/local/opencv-3.4.12/share/OpenCV")
find_package(OpenCV 3.4.12 REQUIRED)
include_directories("usr/local/opencv-3.4.12/include")

find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${g2o_INCLUDE_DIRS} 
        ${Sophus_INCLUDE_DIRS} 
        "/usr/include/eigen3/")

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d g2o_core g2o_stuff ${OpenCV_LIBS} Sophus::Sophus)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d g2o_core g2o_stuff ${OpenCV_LIBS} Sophus::Sophus)

