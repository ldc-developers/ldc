project(tango)

cmake_minimum_required(VERSION 2.6)

#
# Main configuration.
# 

set(INCLUDE_INSTALL_DIR   ${CMAKE_INSTALL_PREFIX}/include/d         CACHE PATH    "Directory where you have Tango header files installed")
set(CMAKE_INSTALL_LIBDIR  ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}  CACHE PATH    "Directory where you want LDC to look for libraries")
set(RUNTIME_AIO tango)
get_directory_property(PROJECT_PARENT_DIR DIRECTORY ${PROJECT_SOURCE_DIR} PARENT_DIRECTORY)

configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}_install.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.conf)
configure_file(${PROJECT_PARENT_DIR}/${LDC_EXE}.rebuild.conf.in ${PROJECT_BINARY_DIR}/../bin/${LDC_EXE}_install.rebuild.conf)
