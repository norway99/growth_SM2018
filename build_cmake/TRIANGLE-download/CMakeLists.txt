cmake_minimum_required(VERSION 2.8.2)

project(TRIANGLE-download NONE)

include(ExternalProject)
ExternalProject_Add(TRIANGLE-download
                    GIT_REPOSITORY;https://github.com/libigl/triangle;GIT_TAG;d284c4a843efac043c310f5fa640b17cf7d96170;UPDATE_DISCONNECTED 1
                    SOURCE_DIR          "/home/aparnank/mvlab/growth_SM2018_forked/build_cmake/TRIANGLE-src"
                    BINARY_DIR          "/home/aparnank/mvlab/growth_SM2018_forked/build_cmake/TRIANGLE-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
