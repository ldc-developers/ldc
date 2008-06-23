project.name = llvmdc

-- options

-- we always make vtables opaque, it simply kills performance...
OPAQUE_VTBLS = 1

-- use of boehm gc
if OS == "windows" then
    USE_BOEHM_GC = 0
else
    addoption("no-boehm", "Disable use of the Boehm GC")

    if options["no-boehm"] then
        USE_BOEHM_GC = 0
    else
        USE_BOEHM_GC = 1
    end
end

-- idgen
package = newpackage()
package.name = "idgen"
package.kind = "exe"
package.language = "c++"
package.files = { "dmd/idgen.c" }
package.buildoptions = { "-x c++" }
package.postbuildcommands = { "./idgen", "mv -f id.c id.h dmd" }

-- impcnvgen
package = newpackage()
package.name = "impcnvgen"
package.kind = "exe"
package.language = "c++"
package.files = { "dmd/impcnvgen.c" }
package.buildoptions = { "-x c++" }
package.postbuildcommands = { "./impcnvgen", "mv -f impcnvtab.c dmd" }

-- llvmdc
package = newpackage()
package.bindir = "bin"
package.name = "llvmdc"
package.kind = "exe"
package.language = "c++"
package.files = { matchfiles("dmd/*.c"), matchfiles("gen/*.cpp"), matchfiles("ir/*.cpp") }
package.excludes = { "dmd/idgen.c", "dmd/impcnvgen.c" }
package.buildoptions = { "-x c++", "`llvm-config --cxxflags`" }
package.linkoptions = {
    -- long but it's faster than just 'all'
    "`llvm-config --libs core asmparser bitreader bitwriter linker support target transformutils scalaropts ipo instrumentation x86 powerpc`",
    "`llvm-config --ldflags`",
}
package.defines = {
    "IN_LLVM",
    "_DH",
    "OPAQUE_VTBLS="..OPAQUE_VTBLS,
    "USE_BOEHM_GC="..USE_BOEHM_GC,
}
package.config.Release.defines = { "LLVMD_NO_LOGGER" }
package.config.Debug.buildoptions = { "-g -O0" }
--package.targetprefix = "llvm"
package.includepaths = { ".", "dmd" }
--package.postbuildcommands = { "cd runtime; ./build.sh; cd .." }
if USE_BOEHM_GC == 1 then
    package.links = { "gc" }
end
