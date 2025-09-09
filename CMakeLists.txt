cmake_minimum_required(VERSION 3.10)
project(perf_model)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(dramsim3)

add_executable(perf_model
        src/main.cc

        src/frontends/standard/StandardLayers.cc
        src/frontends/standard/StandardParser.cc
        src/frontends/standard/StandardArch.cc

        src/units/standard/SysArray.cc
        src/units/standard/VectorUnit.cc

        src/Job.cc
        src/global.cc
        src/perf_enums.cc
        src/State.cc
        src/Arch.cc
        src/memory.cc
        src/EnqueueStructures.cc
        src/NNLayers.cc
)
include_directories(include)

option(USE_VCD "Enable per-cycle VCD dumping of unit states" OFF)

if (${USE_VCD})
    message("VCD Dumping enabled!")
    add_compile_definitions(VCD=1)
endif ()
target_link_libraries(perf_model PRIVATE dramsim3 args)
message(STATUS "PERF_MODEL: ${PERF_MODEL}")