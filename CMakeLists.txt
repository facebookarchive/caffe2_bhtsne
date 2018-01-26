# ---[ An example on how to properly write a CMake file for Caffe2 Module.

# ---[ (1) CMake version requirement.
# This is the common requirement for your cmake version. Usually Caffe2
# requires a minimum of 2.8.12.
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# ---[ (2) Find the Caffe2 package.
# Caffe2 provides cmake config files that give you modern C++ target
# definitions. After find_package, you should be able to use the caffe2
# and caffe2_gpu (depending on whether your Caffe2 installation has gpu)
# in your targets.
#
# The reason we wrap the find_package command inside an if statement is
# that, Caffe2 itself contains a series of modules that is potentially
# built together with the Caffe2 main build. During that time, one does
# not need to invoke the find_package command.
if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  find_package(Caffe2 REQUIRED)
endif()

# ---[ (3) Your main library.
# This is the standard C++ library that you will build for your module.
# Give it a name and include the corresponding source files. We recommend
# you to name the library "caffe2_something", where something is the
# functionality it provides. The reason is also that, in this way, we can
# clearly mark what interface it provides to downstream libraries, such as
# "caffe2_rocksdb".
add_library(caffe2_tsne
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tsne_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bhtsne/sptree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bhtsne/tsne.cpp
    )

target_include_directories(caffe2_tsne
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bhtsne/)

# ---[ (4) Add your custom flags and target rules here.
# For the case of TSNE, we don't have anything to add.

# ---[ (5) Link your library against caffe2.
target_link_libraries(caffe2_tsne caffe2)

# ---[ (6) Install your target library to the intended target folder.
# This allows your library to then connect with 
install(TARGETS caffe2_tsne DESTINATION lib)
