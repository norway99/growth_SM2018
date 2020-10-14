cmake_minimum_required(VERSION 2.8.2)

project(LIBIGL-download NONE)

include(ExternalProject)
ExternalProject_Add(LIBIGL-download
                    GIT_REPOSITORY;https://github.com/libigl/libigl.git;GIT_TAG;7b4a19d;UPDATE_DISCONNECTED 1
                    SOURCE_DIR          "/home/aparnank/mvlab/growth_SM2018_forked/build_cmake/LIBIGL-src"
                    BINARY_DIR          "/home/aparnank/mvlab/growth_SM2018_forked/build_cmake/LIBIGL-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
