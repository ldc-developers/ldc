# Translates linker args for usage in DMD-compatible command-line.
macro(translate_linker_args in_var out_var)
    set(${out_var} "")
    foreach(f IN LISTS "${in_var}")
        if(NOT "${f}" STREQUAL "")
            if(MSVC)
                string(REPLACE "-LIBPATH:" "/LIBPATH:" f ${f})
                list(APPEND ${out_var} "-L${f}")
            else()
                # Work around `-Xcc=-Wl,...` issue for older ldmd2 host compilers.
                if("${D_COMPILER_ID}" STREQUAL "LDMD" AND "${f}" MATCHES "^-Wl,")
                    list(APPEND ${out_var} "-L${f}")
                else()
                    list(APPEND ${out_var} "-Xcc=${f}")
                endif()
            endif()
        endif()
    endforeach()
endmacro()

# Depends on these global variables:
# - D_COMPILER
# - D_COMPILER_ID
# - D_COMPILER_FLAGS
# - DFLAGS_BASE
# - LDC_LINK_MANUALLY
# - D_LINKER_ARGS
# - LDC_ENABLE_PLUGINS
function(build_d_executable target_name output_exe d_src_files compiler_args linker_args extra_compile_deps link_deps compile_separately)
    set(dflags "${D_COMPILER_FLAGS} ${DFLAGS_BASE} ${compiler_args}")
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
        # Default to -linkonce-templates with LDMD host compiler, to speed-up optimization.
        if("${target_name}" STREQUAL "ldc2" AND LDC_ENABLE_PLUGINS)
            # For plugin support we need ldc2's symbols to be global, don't use -linkonce-templates.
        elseif("${D_COMPILER_ID}" STREQUAL "LDMD")
            set(dflags -linkonce-templates ${dflags})
        endif()
        add_custom_command(
            OUTPUT ${object_file}
            COMMAND ${D_COMPILER} -c ${dflags} -of${object_file} ${d_src_files}
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
                COMMAND ${D_COMPILER} -c ${dflags} -of${object_file} ${f}
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
        target_link_libraries(${target_name} ${link_deps} ${linker_args} ${D_LINKER_ARGS})
    else()
        # Use a response file on Windows when compiling separately, in order not to
        # exceed the max command-line length.
        set(objects_args "${object_files}")
        if(WIN32 AND compile_separately)
            string(REPLACE ";" " " objects_args "${object_files}")
            file(WRITE ${output_exe}.rsp ${objects_args})
            set(objects_args "@${output_exe}.rsp")
        endif()

        set(dep_libs "")
        foreach(l ${link_deps})
            list(APPEND dep_libs "-L$<TARGET_LINKER_FILE:${l}>")
        endforeach()

        set(full_linker_args ${CMAKE_EXE_LINKER_FLAGS} ${linker_args})
        translate_linker_args(full_linker_args translated_linker_args)

        # We need to link against the C++ runtime library.
        if(NOT MSVC AND "${D_COMPILER_ID}" STREQUAL "LDMD" AND NOT "${dflags}" MATCHES "(^|;)-gcc=")
            set(translated_linker_args "-gcc=${CMAKE_CXX_COMPILER}" ${translated_linker_args})
        endif()

        # Use an extra custom target as dependency for the executable in
        # addition to the object files directly to improve parallelization.
        # See https://github.com/ldc-developers/ldc/pull/3575.
        add_custom_target(${target_name}_d_objects DEPENDS ${object_files})

        add_custom_command(
            OUTPUT ${output_exe}
            COMMAND ${D_COMPILER} ${dflags} -of${output_exe} ${objects_args} ${dep_libs} ${translated_linker_args}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            DEPENDS ${target_name}_d_objects ${object_files} ${link_deps}
        )
        add_custom_target(${target_name} ALL DEPENDS ${output_exe})
    endif()
endfunction()
