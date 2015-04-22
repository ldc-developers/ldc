#
# Common CPack configuration
#
set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${LDC_VERSION})
set(CPACK_PACKAGE_CONTACT "public@dicebot.lv")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "LDC: LLVM D Compiler")

#
# Debian specifics
#
execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE) 
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
