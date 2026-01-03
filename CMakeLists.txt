cmake_minimum_required(VERSION 3.10)
project(likd-tree LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Option to enable TBB for parallel execution
option(USE_TBB_PARALLEL "Enable TBB for parallel execution in batch queries and background rebuild" ON)

# Find TBB for parallel execution policy support
find_package(TBB QUIET)
if(TBB_FOUND AND USE_TBB_PARALLEL)
	message(STATUS "Found TBB: ${TBB_VERSION}")
	message(STATUS "TBB parallel execution is ENABLED")
	set(LIKD_TREE_USE_TBB ON)
elseif(TBB_FOUND AND NOT USE_TBB_PARALLEL)
	message(STATUS "Found TBB: ${TBB_VERSION}")
	message(STATUS "TBB parallel execution is DISABLED (set USE_TBB_PARALLEL=ON to enable)")
	set(LIKD_TREE_USE_TBB OFF)
else()
	message(STATUS "TBB not found. Parallel execution policies will use sequential execution.")
	set(LIKD_TREE_USE_TBB OFF)
endif()

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
if(LIKD_TREE_USE_TBB)
  target_link_libraries(kdtree INTERFACE TBB::tbb)
  target_compile_definitions(kdtree INTERFACE USE_TBB_PARALLEL)
endif()

# Demo executable (always built)
add_executable(demo 
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
    test/benchmark.cpp
    thirdparty/ikd-Tree/ikd-Tree/ikd_Tree.cpp
  )
  target_include_directories(benchmark PRIVATE thirdparty/ikd-Tree/ikd-Tree)
  target_link_libraries(benchmark kdtree ${PCL_LIBRARIES} Threads::Threads)
else()
  message(STATUS "Benchmark disabled. Use -DBUILD_BENCHMARK=ON to enable.")
endif()

