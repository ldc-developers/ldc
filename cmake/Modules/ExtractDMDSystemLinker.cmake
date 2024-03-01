# Determines the system linker program and default command line arguments
# used by a (CLI-wise, DMD-compatible) D compiler to link executables.
#
# The following variables are read:
#  - D_COMPILER: The D compiler command (i.e., executable) to use.
#
# The following variables are set:
#  - D_LINKER_COMMAND: The system linker command used (gcc, clang, …)
#    internally by ${D_COMPILER}.
#  - D_LINKER_ARGS: The additional command line arguments the ${D_COMPILER}
#    passes to the linker (apart from those specifying the input object file and
#    the output file).
#

# Create a temporary file with an empty main. We do not use the `-main` compiler
# switch as it might behave differently (e.g. some versions of LDC emit the dummy
# module into a separate __main.o file).
set(source_name cmakeExtractDMDSystemLinker)
set(source_file ${CMAKE_BINARY_DIR}/${source_name}.d)
file(WRITE ${source_file} "void main() {}")

# Compile & link the file in verbose mode and capture the compiler's stdout.
set(result_code)
set(stdout)
set(stderr)
if(UNIX)
    separate_arguments(cmdflags UNIX_COMMAND "${D_COMPILER_FLAGS} ${DFLAGS_BASE}")
else()
    separate_arguments(cmdflags WINDOWS_COMMAND "${D_COMPILER_FLAGS} ${DFLAGS_BASE}")
endif()
execute_process(
    COMMAND ${D_COMPILER} ${cmdflags} -v ${source_file}
    RESULT_VARIABLE result_code
    OUTPUT_VARIABLE stdout
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    ERROR_VARIABLE stderr
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(NOT "${result_code}" STREQUAL "0")
    message(FATAL_ERROR "Failed to link empty D program using '${D_COMPILER} ${D_COMPILER_FLAGS} ${DFLAGS_BASE}':\n${stderr}")
endif()

if("${D_COMPILER_ID}" STREQUAL "GDMD")
    # Extract second to last line, which (due to -v) contains the linker command line.
    string(REGEX REPLACE "\n" ";" stderr_lines "${stderr}")
    list(GET stderr_lines -2 linker_line)
    string(REGEX REPLACE "^ +" "" linker_line "${linker_line}")
else()
    # Extract last line, which (due to -v) contains the linker command line.
    string(REGEX REPLACE "\n" ";" stdout_lines "${stdout}")
    list(GET stdout_lines -1 linker_line)
endif()

# Remove object file/output file arguments. This of course makes assumptions on
# the object file names used by the compilers. Another option would be to swallow
# all .o/-o arguments.
string(REPLACE "${source_name}.o" "" linker_line "${linker_line}")
string(REPLACE "-o ${source_name}" "" linker_line "${linker_line}")

# Split up remaining part into executable and arguments.
separate_arguments(linker_line)
list(GET linker_line 0 D_LINKER_COMMAND)
list(REMOVE_AT linker_line 0)

# Fixup known flags with spaces, such as "-target triple" argument, which would be turned
# into "-target;triple" by `separate_arguments`. Repair by merging into one list item.
string(REGEX REPLACE ";-target;([A-Za-z0-9_-]+);" ";-target \\1;" linker_line "${linker_line}")
string(REGEX REPLACE ";-arch;([A-Za-z0-9_-]+);" ";-arch \\1;" linker_line "${linker_line}")

if("${D_COMPILER_ID}" STREQUAL "GDMD")
    # Filter linker arguments for those we know can be safely reused
    set(D_LINKER_ARGS)
    foreach(arg ${linker_line})
        if("${arg}" MATCHES ^-L.*|^-l.*|^-B.*)
            list(APPEND D_LINKER_ARGS "${arg}")
        endif()
    endforeach()
else()
    set(D_LINKER_ARGS ${linker_line})
endif()
