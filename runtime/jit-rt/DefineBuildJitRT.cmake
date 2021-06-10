if(LDC_DYNAMIC_COMPILE)
    file(GLOB LDC_JITRT_D ${JITRT_DIR}/d/ldc/*.d)

    # Choose the correct subfolder depending on the LLVM version
    file(GLOB LDC_JITRT_CXX ${JITRT_DIR}/cpp/*.cpp)
    file(GLOB LDC_JITRT_H ${JITRT_DIR}/cpp/*.h)
    file(GLOB LDC_JITRT_SO_CXX ${JITRT_DIR}/cpp-so/*.cpp)
    file(GLOB LDC_JITRT_SO_H ${JITRT_DIR}/cpp-so/*.h)
    message(STATUS "Use custom passes in jit: ${LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES}")
    if(LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES)
        file(GLOB LDC_JITRT_SO_PASSES_CXX ${CMAKE_SOURCE_DIR}/gen/passes/*.cpp)
        file(GLOB LDC_JITRT_SO_PASSES_H ${CMAKE_SOURCE_DIR}/gen/passes/*.h)
    else()
        set(LDC_JITRT_SO_PASSES_CXX "")
        set(LDC_JITRT_SO_PASSES_H "")
    endif()

    # Set compiler-dependent flags
    if(MSVC)
        # Omit Default Library Name from the library, so it will work with both release and debug builds
        set(JITRT_EXTRA_FLAGS "${JITRT_EXTRA_FLAGS} /Zl")
    else()
        set(JITRT_EXTRA_FLAGS "${JITRT_EXTRA_FLAGS} -fPIC -fvisibility=hidden")
        if(NOT APPLE)
            CHECK_LINK_FLAG("--exclude-libs=ALL" LINKER_ACCEPTS_EXCLUDE_LIBS_ALL)
            if(LINKER_ACCEPTS_EXCLUDE_LIBS_ALL)
                set(JITRT_EXTRA_LDFLAGS "${JITRT_EXTRA_LDFLAGS} -Wl,--exclude-libs=ALL")
            endif()
        endif()
    endif()

    # Sets up the targets for building the D-source jit-rt object files,
    # appending the names of the (bitcode) files to link into the library to
    # outlist_o (outlist_bc).
    macro(compile_jit_rt_D d_flags lib_suffix path_suffix all_at_once outlist_o outlist_bc)
        get_target_suffix("${lib_suffix}" "${path_suffix}" target_suffix)
        dc("${LDC_JITRT_D}"
           "${JITRT_DIR}/d"
           "${d_flags}"
           "${PROJECT_BINARY_DIR}/objects${target_suffix}"
           "OFF" # emit_bc
           "${all_at_once}"
           ""    # single_obj_name
           ${outlist_o}
           ${outlist_bc}
        )
    endmacro()

    function(build_jit_runtime d_flags c_flags ld_flags path_suffix outlist_targets)
        set(asmprinter "asmprinter")
        if(LDC_LLVM_VER LESS 900)
            set(asmprinter "${LLVM_NATIVE_ARCH}asmprinter")
        endif()
        set(jitrt_components core support irreader executionengine passes nativecodegen orcjit target ${LLVM_NATIVE_ARCH}disassembler ${asmprinter})
        llvm_set_libs(JITRT_LIBS libs "${jitrt_components}")

        get_target_suffix("" "${path_suffix}" target_suffix)
        set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

        add_library(ldc-jit-rt-so${target_suffix} SHARED ${LDC_JITRT_SO_CXX} ${LDC_JITRT_SO_H} ${LDC_JITRT_SO_PASSES_CXX} ${LDC_JITRT_SO_PASSES_H})
        set_common_library_properties(ldc-jit-rt-so${target_suffix}
            ldc-jit ${output_path}
            "${c_flags} ${LLVM_CXXFLAGS} ${LDC_CXXFLAGS} ${JITRT_EXTRA_FLAGS}"
            "${ld_flags} ${JITRT_EXTRA_LDFLAGS}"
            ON
        )
        set_target_properties(ldc-jit-rt-so${target_suffix} PROPERTIES LINKER_LANGUAGE CXX)

        if(LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES)
            target_compile_definitions(ldc-jit-rt-so${target_suffix} PRIVATE LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES)
        endif()

        target_include_directories(ldc-jit-rt-so${target_suffix}
            PRIVATE ${CMAKE_SOURCE_DIR}/gen/passes/)

        target_link_libraries(ldc-jit-rt-so${target_suffix} ${LLVM_JITRT_LIBS} ${LLVM_LDFLAGS})

        set(jitrt_d_o "")
        set(jitrt_d_bc "")
        compile_jit_rt_D("-enable-dynamic-compile;${d_flags}" "" "${path_suffix}" "${COMPILE_ALL_D_FILES_AT_ONCE}" jitrt_d_o jitrt_d_bc)

        add_library(ldc-jit-rt${target_suffix} STATIC ${jitrt_d_o} ${LDC_JITRT_CXX} ${LDC_JITRT_H})
        set_common_library_properties(ldc-jit-rt${target_suffix}
            ldc-jit-rt ${output_path}
            "${c_flags} ${LLVM_CXXFLAGS} ${LDC_CXXFLAGS} ${JITRT_EXTRA_FLAGS}"
            "${ld_flags} ${JITRT_EXTRA_LDFLAGS}"
            OFF
        )

        target_link_libraries(ldc-jit-rt${target_suffix} ldc-jit-rt-so${target_suffix})

        list(APPEND ${outlist_targets} "ldc-jit-rt-so${target_suffix}")
        list(APPEND ${outlist_targets} "ldc-jit-rt${target_suffix}")
        set(${outlist_targets} ${${outlist_targets}} PARENT_SCOPE)
    endfunction()

    # Install D interface files
    install(DIRECTORY ${JITRT_DIR}/d/ldc DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.d")
else()
    function(build_jit_runtime d_flags c_flags ld_flags path_suffix outlist_targets)
    endfunction()
endif()
