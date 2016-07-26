# Determines the system linker program and default command line arguments
# used by a DMD-compatible D compiler to link executables.
#
# The following variables are read:
#  - D_COMPILER: The D compiler command (i.e., executable) to use.
#  - D_COMPILER_DMD_COMPAT: Must be TRUE, only compilers with a DMD-compatible
#    command line interface are supported.
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

# Compile the file in verbose mode and capture the compiler's stdout.
set(result_code)
set(stdout)
execute_process(
    COMMAND ${D_COMPILER} ${D_COMPILER_FLAGS} -v ${source_file}
    RESULT_VARIABLE result_code
    OUTPUT_VARIABLE stdout
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(result_code)
	message(FATAL_ERROR "Failed to compile empty program using D compiler '${D_COMPILER}'")
endif()

# Extract last line, which (due to -v) contains the linker command line.
string(REGEX REPLACE "\n" ";" stdout_lines "${stdout}")
list(GET stdout_lines -1 linker_line)

# Remove object file/output file arguments. This of course makes assumptions on
# the object file names used by the compilers. Another option would be to swallow
# all .o/-o arguments.
string(REPLACE "${source_name}.o" "" linker_line "${linker_line}")
string(REPLACE "-o ${source_name}" "" linker_line "${linker_line}")

# Split up remaining part into executable and arguments.
separate_arguments(linker_line)
list(GET linker_line 0 D_LINKER_COMMAND)
list(REMOVE_AT linker_line 0)
set(D_LINKER_ARGS ${linker_line})
