# Depends on these global variables:
# - D_COMPILER
# - D_COMPILER_FLAGS
# - DDMD_DFLAGS
# - DDMD_LFLAGS
# - LDC_LINK_MANUALLY
# - LDC_LINKERFLAG_LIST
# - LDC_TRANSLATED_LINKER_FLAGS
function(build_d_executable target_name output_exe d_src_files compiler_args linker_args extra_compile_deps link_deps compile_separately)
    set(dflags "${D_COMPILER_FLAGS} ${DDMD_DFLAGS}")
    if(UNIX)
      separate_arguments(dflags UNIX_COMMAND "${dflags}")
    else()
      separate_arguments(dflags WINDOWS_COMMAND "${dflags}")
    endif()

    get_filename_component(output_dir ${output_exe} DIRECTORY)

    set(object_files)
    if(NOT compile_separately)
        # Compile all D modules to a single object.
        set(object_file ${PROJECT_BINARY_DIR}/obj/${target_name}${CMAKE_CXX_OUTPUT_EXTENSION})
        add_custom_command(
            OUTPUT ${object_file}
            COMMAND ${D_COMPILER} -c ${dflags} -of${object_file} ${compiler_args} ${d_src_files}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            DEPENDS ${d_src_files} ${extra_compile_deps}
        )
        set(object_files ${object_file})
    else()
        # Compile each D module separately.
        foreach(f ${d_src_files})
            file(RELATIVE_PATH object_file ${PROJECT_SOURCE_DIR} ${f})       # make path relative to PROJECT_SOURCE_DIR
            string(REGEX REPLACE "[/\\\\]" "." object_file "${object_file}") # replace path separators with '.'
            string(REGEX REPLACE "^\\.+" "" object_file "${object_file}")    # strip leading dots (e.g., from original '../dir/file.d' => '...dir.file.d' => 'dir.file.d')
            set(object_file ${PROJECT_BINARY_DIR}/obj/${target_name}/${object_file}${CMAKE_CXX_OUTPUT_EXTENSION})
            add_custom_command(
                OUTPUT ${object_file}
                COMMAND ${D_COMPILER} -c ${dflags} -of${object_file} ${compiler_args} ${f}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                DEPENDS ${f} ${extra_compile_deps}
            )
            list(APPEND object_files ${object_file})
        endforeach()
    endif()

    # Link to an executable.
    if(LDC_LINK_MANUALLY)
        add_executable(${target_name} ${object_files})
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${output_dir}
            LINKER_LANGUAGE          CXX
        )
        target_link_libraries(${target_name} ${linker_args} ${LDC_LINKERFLAG_LIST})
    else()
        # Use a response file on Windows when compiling separately, in order not to
        # exceed the max command-line length.
        set(objects_args "${object_files}")
        if(WIN32 AND compile_separately)
            string(REPLACE ";" " " objects_args "${object_files}")
            file(WRITE ${output_exe}.rsp ${objects_args})
            set(objects_args "@${output_exe}.rsp")
        endif()

        set(translated_linker_args "")
        foreach(f ${linker_args})
            list(APPEND translated_linker_args "-L${f}")
        endforeach()
        add_custom_command(
            OUTPUT ${output_exe}
            COMMAND ${D_COMPILER} ${dflags} ${DDMD_LFLAGS} -of${output_exe} ${objects_args} ${translated_linker_args} ${LDC_TRANSLATED_LINKER_FLAGS}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            DEPENDS ${object_files} ${link_deps}
        )
        add_custom_target(${target_name} ALL DEPENDS ${output_exe})
    endif()
endfunction()
