cmake_minimum_required(VERSION 3.1)
# 3.8 gives us built-in CUDA support; 3.10 gives us OpenGL::EGL
project(CppEGLRenderer)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})


#find_package(pybind11 REQUIRED)
add_subdirectory(pybind11)


find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)

# Search for EGL; nvidia drivers ship the library but not headers, so we redistribute those
# /usr/include (Ubuntu>=18.04) or ./egl_include
find_path(EGL_INCLUDE_DIR NAMES EGL/egl.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/egl_include REQUIRED)
message(STATUS "EGL_INCLUDE_DIR: ${EGL_INCLUDE_DIR}")
find_library(EGL_LIBRARIES NAMES egl EGL REQUIRED)


include_directories(SYSTEM ${EGL_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

cuda_add_library(CppEGLRenderer MODULE  cpp/egl_renderer.cpp)
# Equivalent to pass flags -lGL, -lGLU and -lglut
target_link_libraries(CppEGLRenderer PRIVATE pybind11::module dl pthread GL)
set_target_properties(CppEGLRenderer PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")


set(EGL_HEADER_FILES ${EGL_INCLUDE_DIR}/EGL/egl.h ${EGL_INCLUDE_DIR}/EGL/eglext.h)  # TODO: remove for ubuntu >= 18.04


add_executable(query_devices cpp/query_devices.cpp ${EGL_HEADER_FILES})
target_include_directories(query_devices PRIVATE ${EGL_INCLUDE_DIR})  # TODO: remove for ubuntu >= 18.04

add_executable(test_device cpp/test_device.cpp ${EGL_HEADER_FILES})
target_include_directories(test_device PRIVATE ${EGL_INCLUDE_DIR})  # TODO: remove for ubuntu >= 18.04

target_link_libraries(query_devices GL EGL dl pthread)
target_link_libraries(test_device GL EGL dl pthread)
