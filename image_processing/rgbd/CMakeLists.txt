find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)

set(OpenCV_DIR "/usr/local/opencv-3.4.12/share/OpenCV")
include_directories("usr/local/opencv-3.4.12/include")
find_package(OpenCV 3.4.12 REQUIRED)

add_executable(jointMap jointMap.cpp)
target_link_libraries(jointMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
target_link_libraries(jointMap Sophus::Sophus)
