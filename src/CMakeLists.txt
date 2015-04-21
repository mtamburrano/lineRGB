cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_modules/")

find_package(OpenCV REQUIRED)
find_package(libusb-1.0 REQUIRED)

include_directories (${OpenCV_INCLUDE_DIR} )
include_directories(${LIBUSB_1_INCLUDE_DIRS})

link_directories (${cvimpulso_SOURCE_DIR}/lib)


FILE(GLOB line_rgb_source *.cpp)


ADD_LIBRARY(line_rgb SHARED ${line_rgb_source})

target_link_libraries(line_rgb ${OpenCV_LIBS} boost_system boost_filesystem pthread freenect)

#COMMENTED BECAUSE MULTIPLE DEFINITIONS WITH TRAINER
#/@ TODO: togliere le funzioni in comune con line_trainer
#add_executable (line_test line_test.cpp)
#target_link_libraries(line_test line_rgb )

add_executable (line_trainer trainer.cpp)
target_link_libraries(line_trainer line_rgb )

add_executable (line_camera_rgb utils/line_camera_rgb.cpp)
target_link_libraries(line_camera_rgb line_rgb )

add_executable (generate_similarity_hsv ../utils/generate_similarity_HSV.cpp)
target_link_libraries(generate_similarity_hsv line_rgb )

