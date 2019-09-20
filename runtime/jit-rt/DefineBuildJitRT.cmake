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
        set(JITRT_EXTRA_FLAGS "/Zl")

    else()
        set(JITRT_EXTRA_FLAGS "-fPIC -std=c++11 -fvisibility=hidden")
        if(NOT APPLE)
            CHECK_LINK_FLAG("--exclude-libs=ALL" LINKER_ACCEPTS_EXCLUDE_LIBS_ALL)
            if(LINKER_ACCEPTS_EXCLUDE_LIBS_ALL)
                set(JITRT_EXTRA_LDFLAGS "-Wl,--exclude-libs=ALL")
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
           "OFF"
           "${all_at_once}"
           ${outlist_o}
           ${outlist_bc}
        )
    endmacro()

    function(build_jit_runtime d_flags c_flags ld_flags path_suffix outlist_targets)
        # Jit runtime needs a different set of libraries from compiler and we
        # can't do find_package(LLVM) because we already have one in top-level cmake
        # Also we don't have access to llvm_map_components_to_libnames because we need
        # to do find_package(LLVM CONFIG) for it so here is a hackish way to get it
        include("${LLVM_CMAKEDIR}/LLVMConfig.cmake")
        include("${LLVM_CMAKEDIR}/LLVM-Config.cmake")

        set(asmprinter "asmprinter")
        if(LDC_LLVM_VER LESS 900)
            set(asmprinter "${LLVM_NATIVE_ARCH}asmprinter")
        endif()
        llvm_map_components_to_libnames(JITRT_LLVM_LIBS core support irreader executionengine passes nativecodegen orcjit target
            "${LLVM_NATIVE_ARCH}disassembler" "${asmprinter}")

        foreach(libname ${JITRT_LLVM_LIBS})
            unset(JITRT_TEMP_LIB CACHE)
            find_library(JITRT_TEMP_LIB ${libname} PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
            if(NOT JITRT_TEMP_LIB)
                message(STATUS "lib ${libname} not found, skipping jit runtime build")
                return()
            endif()
            unset(JITRT_TEMP_LIB CACHE)
        endforeach()

        get_target_suffix("" "${path_suffix}" target_suffix)
        set(output_path ${CMAKE_BINARY_DIR}/lib${path_suffix})

        add_library(ldc-jit-rt-so${target_suffix} SHARED ${LDC_JITRT_SO_CXX} ${LDC_JITRT_SO_H} ${LDC_JITRT_SO_PASSES_CXX} ${LDC_JITRT_SO_PASSES_H})
        set_common_library_properties(ldc-jit-rt-so${target_suffix}
            ldc-jit ${output_path}
            "${c_flags} ${LDC_CXXFLAGS} ${LLVM_CXXFLAGS} ${JITRT_EXTRA_FLAGS}"
            "${ld_flags} ${JITRT_EXTRA_LDFLAGS}"
            ON
        )
        set_target_properties(ldc-jit-rt-so${target_suffix} PROPERTIES LINKER_LANGUAGE CXX)

        if(LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES)
            target_compile_definitions(ldc-jit-rt-so${target_suffix} PRIVATE LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES)
        endif()

        target_include_directories(ldc-jit-rt-so${target_suffix}
            PRIVATE ${CMAKE_SOURCE_DIR}/gen/passes/)

        target_link_libraries(ldc-jit-rt-so${target_suffix} ${JITRT_LLVM_LIBS})

        set(jitrt_d_o "")
        set(jitrt_d_bc "")
        compile_jit_rt_D("-enable-dynamic-compile;${d_flags}" "" "${path_suffix}" "${COMPILE_ALL_D_FILES_AT_ONCE}" jitrt_d_o jitrt_d_bc)

        add_library(ldc-jit-rt${target_suffix} STATIC ${jitrt_d_o} ${LDC_JITRT_CXX} ${LDC_JITRT_H})
        set_common_library_properties(ldc-jit-rt${target_suffix}
            ldc-jit-rt ${output_path}
            "${c_flags} ${JITRT_EXTRA_FLAGS}"
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
