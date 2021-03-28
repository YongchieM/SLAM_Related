set(CMAKE_CXX_FLAGS "-std=c++11")
set(OpenCV_DIR "/usr/local/opencv-3.4.12/share/OpenCV")
include_directories("usr/local/opencv-3.4.12/include")

find_package(OpenCV 3.4.12 REQUIRED)

add_executable(imageBasics imageBasics.cpp)
add_executable(undistortImage undistortImage.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})
target_link_libraries(undistortImage ${OpenCV_LIBS})
