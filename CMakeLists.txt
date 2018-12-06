cmake_minimum_required(VERSION 3.8)

project(Multithreading)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_PTHREAD )

add_compile_options(
  "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-Wsuggest-override" "-Wnon-virtual-dtor" "-O0" "-g3" "-ggdb" "-pthread" 
)

set(SOURCE_FILES multithreading.cpp)

add_executable(multithreading ${SOURCE_FILES})

target_link_libraries(multithreading pthread)