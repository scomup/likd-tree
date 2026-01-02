cmake_minimum_required(VERSION 3.10)
project(likd-tree LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(PCL QUIET COMPONENTS common io)
if(PCL_FOUND)
	message(STATUS "Found PCL: ${PCL_VERSION}")
	include_directories(${PCL_INCLUDE_DIRS})
	link_directories(${PCL_LIBRARY_DIRS})
	add_definitions(${PCL_DEFINITIONS})
else()
	message(STATUS "PCL not found; test will build with a lightweight Point struct unless PCL is available.")
endif()

add_library(kdtree INTERFACE)
target_include_directories(kdtree INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Demo executable (always built)
add_executable(demo 
src/likd_tree.cpp
test/demo.cpp
)
target_link_libraries(demo kdtree ${PCL_LIBRARIES})
find_package(Threads REQUIRED)
target_link_libraries(demo Threads::Threads)

# Benchmark executable (only built with -DBUILD_BENCHMARK=ON)
option(BUILD_BENCHMARK "Build benchmark comparing likd-tree vs ikd-tree" OFF)
if(BUILD_BENCHMARK)
  message(STATUS "Building benchmark executable")
  
  # Automatically download ikd-Tree if not available
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ikd-Tree/ikd-Tree/ikd_Tree.h")
    find_package(Git QUIET)
    if(GIT_FOUND)
      message(STATUS "Attempting to initialize ikd-Tree submodule...")
      execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMOD_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
      )
    endif()
    
    # If submodule init failed or git not available, use FetchContent
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ikd-Tree/ikd-Tree/ikd_Tree.h")
      message(STATUS "Submodule not available, downloading ikd-Tree using FetchContent...")
      include(FetchContent)
      FetchContent_Declare(
        ikd-tree
        GIT_REPOSITORY https://github.com/hku-mars/ikd-Tree.git
        GIT_TAG main
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ikd-Tree
      )
      FetchContent_MakeAvailable(ikd-tree)
    else()
      message(STATUS "ikd-Tree submodule initialized successfully")
    endif()
  endif()
  
  add_executable(benchmark 
    src/likd_tree.cpp
    test/benchmark.cpp
    thirdparty/ikd-Tree/ikd-Tree/ikd_Tree.cpp
  )
  target_include_directories(benchmark PRIVATE thirdparty/ikd-Tree/ikd-Tree)
  target_link_libraries(benchmark kdtree ${PCL_LIBRARIES} Threads::Threads)
else()
  message(STATUS "Benchmark disabled. Use -DBUILD_BENCHMARK=ON to enable.")
endif()

