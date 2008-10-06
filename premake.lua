project.name = ldc

-- options

-- we always make vtables opaque, it simply kills performance...
OPAQUE_VTBLS = 1

-- use of boehm gc
USE_BOEHM_GC = 0
if OS ~= "windows" then
    addoption("enable-boehm-gc", "Enable use of the Boehm GC (broken!)")

    if options["enable-boehm-gc"] then
        USE_BOEHM_GC = 1
    end
end

-- are we on a Posix system?
POSIX = 1
if OS == "windows" then
    POSIX = 0
end

-- D version - don't change these !!!
DMDV1 = "1"

-- idgen
package = newpackage()
package.name = "idgen"
package.kind = "exe"
package.language = "c++"
package.files = { "dmd/idgen.c" }
package.buildoptions = { "-x c++" }
package.postbuildcommands = { "./idgen", "mv -f id.c id.h dmd" }
package.defines = { "DMDV1="..DMDV1 }

-- impcnvgen
package = newpackage()
package.name = "impcnvgen"
package.kind = "exe"
package.language = "c++"
package.files = { "dmd/impcnvgen.c" }
package.buildoptions = { "-x c++" }
package.postbuildcommands = { "./impcnvgen", "mv -f impcnvtab.c dmd" }
package.defines = { "DMDV1="..DMDV1 }

-- ldc
package = newpackage()
package.bindir = "bin"
package.name = "ldc"
package.kind = "exe"
package.language = "c++"
package.files = { matchfiles("dmd/*.c"), matchfiles("gen/*.cpp"), matchfiles("ir/*.cpp") }
package.excludes = { "dmd/idgen.c", "dmd/impcnvgen.c" }
package.buildoptions = { "-x c++", "`llvm-config --cxxflags`" }
package.linkoptions = {
    -- long but it's faster than just 'all'
    "`llvm-config --libs bitwriter linker ipo instrumentation`",
    "`llvm-config --ldflags`",
}
package.defines = {
    "IN_LLVM",
    "_DH",
    "OPAQUE_VTBLS="..OPAQUE_VTBLS,
    "USE_BOEHM_GC="..USE_BOEHM_GC,
    "DMDV1="..DMDV1,
    "POSIX="..POSIX,
}
package.config.Release.defines = { "LLVMD_NO_LOGGER" }
package.config.Debug.buildoptions = { "-g -O0" }
--package.targetprefix = "llvm"
package.includepaths = { ".", "dmd" }
--package.postbuildcommands = { "cd runtime; ./build.sh; cd .." }
if USE_BOEHM_GC == 1 then
    package.links = { "gc" }
end
