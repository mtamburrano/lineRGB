cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIR} )

link_directories (${cvimpulso_SOURCE_DIR}/lib)


FILE(GLOB line_rgb_source *.cpp)


ADD_LIBRARY(line_rgb SHARED ${line_rgb_source})

target_link_libraries(line_rgb ${OpenCV_LIBS})

add_executable (line_test line_test.cpp)
target_link_libraries(line_test line_rgb )

add_executable (line_camera_rgb utils/line_camera_rgb.cpp)
target_link_libraries(line_camera_rgb line_rgb )

