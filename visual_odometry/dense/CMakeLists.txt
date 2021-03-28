project(ch8)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

set(OpenCV_DIR "/usr/local/opencv4/share/opencv4")
find_package(OpenCV 4.5.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS} "/usr/include/eigen3/" ${Pangolin_INCLUDE__DIRS} "usr/local/opencv4/include")
                    
add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Sophus::Sophus)

