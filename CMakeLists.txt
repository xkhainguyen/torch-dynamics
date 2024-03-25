cmake_minimum_required(VERSION 3.10.1)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++14")

# Find Eigen3
find_package(Eigen3)

# Find Pinocchio and Casadi
find_package(PkgConfig REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)
pkg_check_modules(casadi REQUIRED casadi)

### ----- Create symbolic generation exe ----- ###
project(gen_symb)

include_directories(
        include
        ${EIGEN3_INCLUDE_DIRS}
        ${pinocchio_INCLUDE_DIRS}
        ${casadi_INCLUDE_DIRS}
)
link_directories(
        ${pinocchio_LIBRARY_DIRS}
        ${casadi_LIBRARY_DIRS}
)

add_executable(${PROJECT_NAME} gen_symbolic.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC)

target_link_libraries(${PROJECT_NAME}
        ${pinocchio_LIBRARIES}
        ${casadi_LIBRARIES}
)

target_compile_options(${PROJECT_NAME} PUBLIC)

### ----- Create shared library ----- ###
project(dynamics)

include_directories(
        include
)

add_library(${PROJECT_NAME} SHARED example.cpp
        forward_dynamics.c)

target_include_directories(${PROJECT_NAME} PUBLIC)

target_compile_options(${PROJECT_NAME} PUBLIC)