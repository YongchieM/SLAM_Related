project(stereoVision)

set(OpenCV_DIR "/usr/local/opencv-3.4.12/share/OpenCV")
include_directories("usr/local/opencv-3.4.12/include")
find_package(OpenCV 3.4.12 REQUIRED)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
