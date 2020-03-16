# - Preset cross-compilation configurations for C/ASM and D compilation
# and linking, for supported targets
#
# This module sets compiler flags for a few C and assembly files in
# DRuntime and Phobos, and linker flags to link the standard library as
# a shared library and build the test runners, for various
# cross-compilation targets that the LDC developers have tried out.
#
# It is enabled by setting LDC_TARGET_PRESET to a supported platform,
# after which the appropriate TARGET_SYSTEM is set and a target triple
# is appended to D_FLAGS.
#
# You can pass in custom RT_CFLAGS and LD_FLAGS of your choosing, but
# if they're left unconfigured, they will also be set to sensible
# defaults.

if(NOT LDC_TARGET_PRESET STREQUAL "")
    if(LDC_TARGET_PRESET MATCHES "Android")
        set(ANDROID_API "21")
    endif()
    # This initial RT_CFLAGS/LD_FLAGS configuration for Android is a
    # convenience for natively compiling, because CMake cannot detect
    # Android as a separate platform from Linux.
    if(RT_CFLAGS STREQUAL "" AND LDC_TARGET_PRESET MATCHES "Android")
        set(RT_CFLAGS_UNCONFIGURED True)
        set(RT_CFLAGS "-ffunction-sections -fdata-sections -funwind-tables -fstack-protector-strong -Wno-invalid-command-line-argument -Wno-unused-command-line-argument -no-canonical-prefixes -g -DNDEBUG -DANDROID  -D__ANDROID_API__=${ANDROID_API} -Wa,--noexecstack -Wformat -Werror=format-security")

        if(LDC_TARGET_PRESET MATCHES "arm")
            append("-target armv7-none-linux-androideabi${ANDROID_API} -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -mthumb -Oz" RT_CFLAGS)
        elseif(LDC_TARGET_PRESET MATCHES "aarch64")
            append("-target aarch64-none-linux-android -O2" RT_CFLAGS)
        elseif(LDC_TARGET_PRESET MATCHES "x86")
            append("-target i686-none-linux-android -O2 -mstackrealign" RT_CFLAGS)
        elseif(LDC_TARGET_PRESET MATCHES "x64")
            append("-target x86_64-none-linux-android${ANDROID_API} -O2" RT_CFLAGS)
        endif()
    endif()

    if(LD_FLAGS STREQUAL "" AND LDC_TARGET_PRESET MATCHES "Android")
        set(LD_FLAGS_UNCONFIGURED True)
        set(LD_FLAGS "-Wl,--gc-sections -Wl,-z,nocopyreloc -no-canonical-prefixes -Wl,--no-undefined -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -Wl,--warn-shared-textrel -Wl,--fatal-warnings -fpie -pie")

        if(LDC_TARGET_PRESET MATCHES "arm")
            append("-target armv7-none-linux-androideabi${ANDROID_API} -Wl,--fix-cortex-a8" LD_FLAGS)
        elseif(LDC_TARGET_PRESET MATCHES "aarch64")
            append("-target aarch64-none-linux-android" LD_FLAGS)
        elseif(LDC_TARGET_PRESET MATCHES "x86")
            append("-target i686-none-linux-android${ANDROID_API}" LD_FLAGS)
        elseif(LDC_TARGET_PRESET MATCHES "x64")
            append("-target x86_64-none-linux-android${ANDROID_API}" LD_FLAGS)
        endif()
    endif()

    if(LDC_TARGET_PRESET MATCHES "Windows")
        set(TARGET_SYSTEM "Windows;MSVC")
        if(LDC_TARGET_PRESET MATCHES "x64")
            # stub example, fill in with the rest
            list(APPEND D_FLAGS "-mtriple=x86_64-pc-windows-msvc")
        endif()
    elseif(LDC_TARGET_PRESET MATCHES "Android")
        set(TARGET_SYSTEM "Android;Linux;UNIX")

        if(LDC_TARGET_PRESET MATCHES "arm")
            set(TARGET_ARCH "arm")
            set(LLVM_TARGET_TRIPLE "armv7-none-linux-android")
            set(TOOLCHAIN "arm-linux-androideabi")
            set(TOOLCHAIN_TARGET_TRIPLE "arm-linux-androideabi")
        elseif(LDC_TARGET_PRESET MATCHES "aarch64")
            set(TARGET_ARCH "arm64")
            set(LLVM_TARGET_TRIPLE "aarch64-none-linux-android")
            set(TOOLCHAIN "aarch64-linux-android")
            set(TOOLCHAIN_TARGET_TRIPLE "aarch64-linux-android")
        elseif(LDC_TARGET_PRESET MATCHES "x86")
            set(TARGET_ARCH "x86")
            set(LLVM_TARGET_TRIPLE "i686-none-linux-android")
            set(TOOLCHAIN "x86")
            set(TOOLCHAIN_TARGET_TRIPLE "i686-linux-android")
        elseif(LDC_TARGET_PRESET MATCHES "x64")
            set(TARGET_ARCH "x86_64")
            set(LLVM_TARGET_TRIPLE "x86_64-none-linux-android")
            set(TOOLCHAIN "x86_64")
            set(TOOLCHAIN_TARGET_TRIPLE "x86_64-linux-android")
        else()
            message(FATAL_ERROR "Android platform ${LDC_TARGET_PRESET} is not supported.")
        endif()
        list(APPEND D_FLAGS "-mtriple=${LLVM_TARGET_TRIPLE}")

        # Check if we're using the NDK by looking for the toolchains
        # directory in CC
        if(CMAKE_C_COMPILER MATCHES "toolchains")
            # Extract the NDK path and platform from CC
            string(REGEX REPLACE ".toolchains.+" "" NDK_PATH ${CMAKE_C_COMPILER})
            string(REGEX REPLACE ".+/prebuilt/([^/]+)/.+" "\\1" NDK_HOST_PLATFORM ${CMAKE_C_COMPILER})
            set(TOOLCHAIN_VERSION "4.9")

            if(RT_CFLAGS_UNCONFIGURED)
                append("-gcc-toolchain ${NDK_PATH}/toolchains/${TOOLCHAIN}-${TOOLCHAIN_VERSION}/prebuilt/${NDK_HOST_PLATFORM} --sysroot ${NDK_PATH}/sysroot -isystem ${NDK_PATH}/sysroot/usr/include/${TOOLCHAIN_TARGET_TRIPLE}" RT_CFLAGS)

                if(LDC_TARGET_PRESET MATCHES "arm")
                    append("-fno-integrated-as" RT_CFLAGS)
                endif()
            endif()

            if(LD_FLAGS_UNCONFIGURED)
                set(LD_BFD "-fuse-ld=bfd")
                # work around Windows bug, android-ndk/ndk#75
                if(NDK_HOST_PLATFORM MATCHES "windows")
                    set(LD_BFD "${LD_BFD}.exe")
                endif()

                append("--sysroot=${NDK_PATH}/platforms/android-${ANDROID_API}/arch-${TARGET_ARCH} -gcc-toolchain ${NDK_PATH}/toolchains/${TOOLCHAIN}-${TOOLCHAIN_VERSION}/prebuilt/${NDK_HOST_PLATFORM} ${LD_BFD}" LD_FLAGS)
            endif()
        endif()
    elseif(LDC_TARGET_PRESET MATCHES "PlayStation4")
        set(LLVM_TARGET_TRIPLE "x86_64-scei-ps4")
        list(APPEND D_FLAGS "-mtriple=${LLVM_TARGET_TRIPLE}")
    else()
        message(FATAL_ERROR "LDC_TARGET_PRESET ${LDC_TARGET_PRESET} is not supported yet, pull requests to add common flags are welcome.")
    endif()
endif()
