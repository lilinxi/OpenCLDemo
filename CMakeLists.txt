cmake_minimum_required(VERSION 3.15)
project(OpenCLDemo)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
message(STATUS "OpenCL library status:")
message(STATUS "    found: ${OpenCL_FOUND}")
message(STATUS "    include path: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "    library: ${OpenCL_LIBRARIES}")
message(STATUS "    version: ${OpenCL_VERSION_STRING}")
message(STATUS "    major version: ${OpenCL_VERSION_MAJOR}")
message(STATUS "    minor version: ${OpenCL_VERSION_MINOR}")

add_executable(OpenCLDemo main.cpp)
target_link_libraries(OpenCLDemo ${OpenCL_LIBRARIES})

add_executable(helloworld helloworld.cpp)
target_link_libraries(helloworld ${OpenCL_LIBRARIES})
