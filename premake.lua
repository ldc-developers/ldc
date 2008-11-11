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


-- guess the host machine description
-- also allow overriding it

addoption("target-override", "Override the default target machine");

TRIPLE = "";
if options["target-override"] then
    TRIPLE = options["target-override"]
else
    os.execute("sh config.guess > default-target-triple.tmp")
    TRIPLE = io.open("default-target-triple.tmp"):read()
end

io.write("Default target: '"..TRIPLE.."'\n");

-- x86 ABI support
X86_REVERSE_PARAMS = 1
X86_PASS_IN_EAX = 1

-- D version
DMDV2 = true

if DMDV2 then
    DMD_V_DEF = "DMDV2=1"
    DMD_DIR = "dmd2"
else
    DMD_V_DEF = "DMDV1=1"
    DMD_DIR = "dmd"
end

-- idgen
package = newpackage()
package.name = "idgen"
package.kind = "exe"
package.language = "c++"
package.files = { DMD_DIR.."/idgen.c" }
package.buildoptions = { "-x c++" }
package.postbuildcommands = { "./idgen", "mv -f id.c id.h "..DMD_DIR }
package.defines = { DMD_V_DEF }

-- impcnvgen
package = newpackage()
package.name = "impcnvgen"
package.kind = "exe"
package.language = "c++"
package.files = { DMD_DIR.."/impcnvgen.c" }
package.buildoptions = { "-x c++" }
package.postbuildcommands = { "./impcnvgen", "mv -f impcnvtab.c "..DMD_DIR }
package.defines = { DMD_V_DEF }

-- ldc
package = newpackage()
package.bindir = "bin"
package.name = DMDV2 and "ldc2" or "ldc"
package.kind = "exe"
package.language = "c++"
package.files = { matchfiles(DMD_DIR.."/*.c"), matchfiles("gen/*.cpp"), matchfiles("ir/*.cpp") }
package.excludes = { DMD_DIR.."/idgen.c", DMD_DIR.."/impcnvgen.c" }
package.buildoptions = { "-x c++", "`llvm-config --cxxflags`" }
package.linkoptions = {
    -- long but it's faster than just 'all'
    "`llvm-config --libs bitwriter linker ipo instrumentation backend`",
    "`llvm-config --ldflags`",
}
package.defines = {
    "IN_LLVM",
    "_DH",
    DMD_V_DEF,
    "OPAQUE_VTBLS="..OPAQUE_VTBLS,
    "USE_BOEHM_GC="..USE_BOEHM_GC,
    "POSIX="..POSIX,
    "DEFAULT_TARGET_TRIPLE=\\\""..TRIPLE.."\\\"",
    "X86_REVERSE_PARAMS="..X86_REVERSE_PARAMS,
    "X86_PASS_IN_EAX="..X86_PASS_IN_EAX,
}

package.config.Release.defines = { "LLVMD_NO_LOGGER" }
package.config.Debug.buildoptions = { "-g -O0" }

--package.targetprefix = "llvm"
package.includepaths = { ".", DMD_DIR }

--package.postbuildcommands = { "cd runtime; ./build.sh; cd .." }

if USE_BOEHM_GC == 1 then
    package.links = { "gc" }
end
