cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pcb)
add_definitions(-std=c++14)
set(CMAKE_PREFIX_PATH "/home/agent/xcj_ws/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV 3 REQUIRED)

if(OpenCV_FOUND)
	message(STATUS "OpenCV version: "${OpenCV_VERSION})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pcb demo.cpp)
target_link_libraries(pcb ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET pcb PROPERTY CXX_STANDARD 14)

add_executable(imagecorp corpImage.cpp)
target_link_libraries(imagecorp ${OpenCV_LIBS})