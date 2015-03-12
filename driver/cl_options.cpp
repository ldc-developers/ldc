//===-- cl_options.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/cl_options.h"
#include "mars.h"
#include "gen/cl_helpers.h"
#include "gen/logger.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/DataLayout.h"
#elif LDC_LLVM_VER == 302
#include "llvm/DataLayout.h"
#else
#include "llvm/Target/TargetData.h"
#endif

namespace opts {

// Positional options first, in order:
cl::list<std::string> fileList(
    cl::Positional, cl::desc("files"));

cl::list<std::string> runargs("run",
    cl::desc("Runs the resulting program, passing the remaining arguments to it"),
    cl::Positional,
    cl::PositionalEatsArgs);

static cl::opt<ubyte, true> useDeprecated(
    cl::desc("Allow deprecated code/language features:"),
    cl::ZeroOrMore,
    cl::values(
        clEnumValN(0, "de", "Do not allow deprecated features"),
        clEnumValN(1, "d", "Silently allow deprecated features"),
        clEnumValN(2, "dw", "Warn about the use of deprecated features"),
        clEnumValEnd),
    cl::location(global.params.useDeprecated),
    cl::init(2));

cl::opt<bool, true> enforcePropertySyntax("property",
    cl::desc("Enforce property syntax"),
    cl::ZeroOrMore,
    cl::location(global.params.enforcePropertySyntax));

cl::opt<bool> compileOnly("c",
    cl::desc("Do not link"),
    cl::ZeroOrMore);

cl::opt<bool> createStaticLib("lib",
    cl::desc("Create static library"),
    cl::ZeroOrMore);

cl::opt<bool> createSharedLib("shared",
    cl::desc("Create shared library"),
    cl::ZeroOrMore);

static cl::opt<bool, true> verbose("v",
    cl::desc("Verbose"),
    cl::ZeroOrMore,
    cl::location(global.params.verbose));

static cl::opt<bool, true> verbose_cg("v-cg",
    cl::desc("Verbose codegen"),
    cl::ZeroOrMore,
    cl::location(global.params.verbose_cg));

static cl::opt<ubyte, true> warnings(
    cl::desc("Warnings:"),
    cl::ZeroOrMore,
    cl::values(
        clEnumValN(1, "w",  "Enable warnings"),
        clEnumValN(2, "wi", "Enable informational warnings"),
        clEnumValEnd),
    cl::location(global.params.warnings),
    cl::init(0));

static cl::opt<bool, true> ignoreUnsupportedPragmas("ignore",
    cl::desc("Ignore unsupported pragmas"),
    cl::ZeroOrMore,
    cl::location(global.params.ignoreUnsupportedPragmas));

static cl::opt<ubyte, true> debugInfo(
    cl::desc("Generating debug information:"),
    cl::ZeroOrMore,
    cl::values(
        clEnumValN(1, "g",  "Generate debug information"),
        clEnumValN(2, "gc", "Same as -g, but pretend to be C"),
        clEnumValEnd),
    cl::location(global.params.symdebug),
    cl::init(0));

cl::opt<bool> noAsm("noasm",
    cl::desc("Disallow use of inline assembler"));

// Output file options
cl::opt<bool> dontWriteObj("o-",
    cl::desc("Do not write object file"));

cl::opt<std::string> objectFile("of",
    cl::value_desc("filename"),
    cl::Prefix,
    cl::desc("Use <filename> as output file name"));

cl::opt<std::string> objectDir("od",
    cl::value_desc("objdir"),
    cl::Prefix,
    cl::desc("Write object files to directory <objdir>"));

cl::opt<std::string> soname("soname",
    cl::value_desc("soname"),
    cl::Hidden,
    cl::Prefix,
    cl::desc("Use <soname> as output shared library soname"));


// Output format options
cl::opt<bool> output_bc("output-bc",
    cl::desc("Write LLVM bitcode"));

cl::opt<bool> output_ll("output-ll",
    cl::desc("Write LLVM IR"));

cl::opt<bool> output_s("output-s",
    cl::desc("Write native assembly"));

cl::opt<cl::boolOrDefault> output_o("output-o",
    cl::desc("Write native object"));

// Disabling Red Zone
cl::opt<bool, true> disableRedZone("disable-red-zone",
  cl::desc("Do not emit code that uses the red zone."),
  cl::location(global.params.disableRedZone),
  cl::init(false));

// DDoc options
static cl::opt<bool, true> doDdoc("D",
    cl::desc("Generate documentation"),
    cl::location(global.params.doDocComments));

cl::opt<std::string> ddocDir("Dd",
    cl::desc("Write documentation file to <docdir> directory"),
    cl::value_desc("docdir"),
    cl::Prefix);

cl::opt<std::string> ddocFile("Df",
    cl::desc("Write documentation file to <filename>"),
    cl::value_desc("filename"),
    cl::Prefix);

// Json options
static cl::opt<bool, true> doJson("X",
    cl::desc("Generate JSON file"),
    cl::location(global.params.doJsonGeneration));

cl::opt<std::string> jsonFile("Xf",
    cl::desc("Write JSON file to <filename>"),
    cl::value_desc("filename"),
    cl::Prefix);

// Header generation options
static cl::opt<bool, true> doHdrGen("H",
    cl::desc("Generate 'header' file"),
    cl::location(global.params.doHdrGeneration));

cl::opt<std::string> hdrDir("Hd",
    cl::desc("Write 'header' file to <hdrdir> directory"),
    cl::value_desc("hdrdir"),
    cl::Prefix);

cl::opt<std::string> hdrFile("Hf",
    cl::desc("Write 'header' file to <filename>"),
    cl::value_desc("filename"),
    cl::Prefix);

static cl::opt<bool, true> hdrKeepAllBodies("Hkeep-all-bodies",
    cl::desc("Keep all function bodies in .di files"),
    cl::ZeroOrMore,
    cl::location(global.params.hdrKeepAllBodies));

static cl::opt<bool, true> unittest("unittest",
    cl::desc("Compile in unit tests"),
    cl::location(global.params.useUnitTests));


static StringsAdapter strImpPathStore("J", global.params.fileImppath);
static cl::list<std::string, StringsAdapter> stringImportPaths("J",
    cl::desc("Where to look for string imports"),
    cl::value_desc("path"),
    cl::location(strImpPathStore),
    cl::Prefix);

static cl::opt<bool, true> addMain("main",
    cl::desc("Add empty main() (e.g. for unittesting)"),
    cl::ZeroOrMore,
    cl::location(global.params.addMain));


// -d-debug is a bit messy, it has 3 modes:
// -d-debug=ident, -d-debug=level and -d-debug (without argument)
// That last of these must be acted upon immediately to ensure proper
// interaction with other options, so it needs some special handling:
std::vector<std::string> debugArgs;

struct D_DebugStorage {
    void push_back(const std::string& str) {
        if (str.empty()) {
            // Bare "-d-debug" has a special meaning.
            global.params.useAssert = true;
            global.params.useArrayBounds = 2;
            global.params.useInvariants = true;
            global.params.useIn = true;
            global.params.useOut = true;
            debugArgs.push_back("1");
        } else {
            debugArgs.push_back(str);
        }
    }
};

static D_DebugStorage dds;

// -debug is already declared in LLVM (at least, in debug builds),
// so we need to be a bit more verbose.
static cl::list<std::string, D_DebugStorage> debugVersionsOption("d-debug",
    cl::desc("Compile in debug code >= <level> or identified by <idents>."),
    cl::value_desc("level/idents"),
    cl::location(dds),
    cl::CommaSeparated,
    cl::ValueOptional);



// -version is also declared in LLVM, so again we need to be a bit more verbose.
cl::list<std::string> versions("d-version",
    cl::desc("Compile in version code >= <level> or identified by <idents>"),
    cl::value_desc("level/idents"),
    cl::CommaSeparated);


static StringsAdapter linkSwitchStore("L", global.params.linkswitches);
static cl::list<std::string, StringsAdapter> linkerSwitches("L",
    cl::desc("Pass <linkerflag> to the linker"),
    cl::value_desc("linkerflag"),
    cl::location(linkSwitchStore),
    cl::Prefix);


cl::opt<std::string> moduleDepsFile("deps",
    cl::desc("Write module dependencies to filename"),
    cl::value_desc("filename"));


cl::opt<std::string> mArch("march",
    cl::desc("Architecture to generate code for:"));

cl::opt<bool> m32bits("m32",
    cl::desc("32 bit target"),
    cl::ZeroOrMore);

cl::opt<bool> m64bits("m64",
    cl::desc("64 bit target"),
    cl::ZeroOrMore);

cl::opt<std::string> mCPU("mcpu",
    cl::desc("Target a specific cpu type (-mcpu=help for details)"),
    cl::value_desc("cpu-name"),
    cl::init(""));

cl::list<std::string> mAttrs("mattr",
    cl::CommaSeparated,
    cl::desc("Target specific attributes (-mattr=help for details)"),
    cl::value_desc("a1,+a2,-a3,..."));

cl::opt<std::string> mTargetTriple("mtriple",
    cl::desc("Override target triple"));

cl::opt<llvm::Reloc::Model> mRelocModel("relocation-model",
    cl::desc("Relocation model"),
    cl::init(llvm::Reloc::Default),
    cl::values(
        clEnumValN(llvm::Reloc::Default, "default",
                   "Target default relocation model"),
        clEnumValN(llvm::Reloc::Static, "static",
                   "Non-relocatable code"),
        clEnumValN(llvm::Reloc::PIC_, "pic",
                   "Fully relocatable, position independent code"),
        clEnumValN(llvm::Reloc::DynamicNoPIC, "dynamic-no-pic",
                   "Relocatable external references, non-relocatable code"),
        clEnumValEnd));

cl::opt<llvm::CodeModel::Model> mCodeModel("code-model",
    cl::desc("Code model"),
    cl::init(llvm::CodeModel::Default),
    cl::values(
        clEnumValN(llvm::CodeModel::Default, "default", "Target default code model"),
        clEnumValN(llvm::CodeModel::Small, "small", "Small code model"),
        clEnumValN(llvm::CodeModel::Kernel, "kernel", "Kernel code model"),
        clEnumValN(llvm::CodeModel::Medium, "medium", "Medium code model"),
        clEnumValN(llvm::CodeModel::Large, "large", "Large code model"),
        clEnumValEnd));

cl::opt<FloatABI::Type> mFloatABI("float-abi",
    cl::desc("ABI/operations to use for floating-point types:"),
    cl::init(FloatABI::Default),
    cl::values(
        clEnumValN(FloatABI::Default, "default", "Target default floating-point ABI"),
        clEnumValN(FloatABI::Soft, "soft", "Software floating-point ABI and operations"),
        clEnumValN(FloatABI::SoftFP, "softfp", "Soft-float ABI, but hardware floating-point instructions"),
        clEnumValN(FloatABI::Hard, "hard", "Hardware floating-point ABI and instructions"),
        clEnumValEnd));

cl::opt<bool> disableFpElim("disable-fp-elim",
              cl::desc("Disable frame pointer elimination optimization"),
              cl::init(false));

static cl::opt<bool, true, FlagParser> asserts("asserts",
    cl::desc("(*) Enable assertions"),
    cl::value_desc("bool"),
    cl::location(global.params.useAssert),
    cl::init(true));

BoundsCheck boundsCheck = BC_Default;

class BoundsChecksAdapter {
public:
    void operator=(bool val) {
        boundsCheck = (val ? BC_On : BC_Off);
    }
};

cl::opt<BoundsChecksAdapter, false, FlagParser> boundsChecksOld("boundscheck",
    cl::desc("(*) Enable array bounds check (deprecated, use -boundscheck=on|off)"));

cl::opt<BoundsCheck, true> boundsChecksNew("boundscheck",
    cl::desc("(*) Enable array bounds check"),
    cl::location(boundsCheck),
    cl::values(
        clEnumValN(BC_Off, "off", "no array bounds checks"),
        clEnumValN(BC_SafeOnly, "safeonly", "array bounds checks for safe functions only"),
        clEnumValN(BC_On, "on", "array bounds checks for all functions"),
        clEnumValEnd));

static cl::opt<bool, true, FlagParser> invariants("invariants",
    cl::desc("(*) Enable invariants"),
    cl::location(global.params.useInvariants),
    cl::init(true));

static cl::opt<bool, true, FlagParser> preconditions("preconditions",
    cl::desc("(*) Enable function preconditions"),
    cl::location(global.params.useIn),
    cl::init(true));

static cl::opt<bool, true, FlagParser> postconditions("postconditions",
    cl::desc("(*) Enable function postconditions"),
    cl::location(global.params.useOut),
    cl::init(true));


static MultiSetter ContractsSetter(false,
    &global.params.useIn, &global.params.useOut, NULL);
static cl::opt<MultiSetter, true, FlagParser> contracts("contracts",
    cl::desc("(*) Enable function pre- and post-conditions"),
    cl::location(ContractsSetter));

bool nonSafeBoundsChecks = true;
static MultiSetter ReleaseSetter(true, &global.params.useAssert,
    &nonSafeBoundsChecks, &global.params.useInvariants,
    &global.params.useOut, &global.params.useIn, NULL);
static cl::opt<MultiSetter, true, cl::parser<bool> > release("release",
    cl::desc("Disables asserts, invariants, contracts and boundscheck"),
    cl::location(ReleaseSetter),
    cl::ValueDisallowed);

cl::opt<bool, true> singleObj("singleobj",
    cl::desc("Create only a single output object file"),
    cl::location(global.params.singleObj));

cl::opt<bool> linkonceTemplates("linkonce-templates",
    cl::desc("Use linkonce_odr linkage for template symbols instead of weak_odr"),
    cl::ZeroOrMore);

cl::opt<bool> disableLinkerStripDead("disable-linker-strip-dead",
    cl::desc("Do not try to remove unused symbols during linking"),
    cl::init(false));

cl::opt<bool> ffastmath("ffast-math",
    cl::desc("Enable unsafe floating-point math"),
    cl::init(false));

cl::opt<bool, true> allinst("allinst",
    cl::desc("generate code for all template instantiations"),
    cl::location(global.params.allInst));

cl::opt<unsigned, true> nestedTemplateDepth("template-depth",
    cl::desc("(experimental) set maximum number of nested template instantiations"),
    cl::location(global.params.nestedTmpl),
    cl::init(500));

cl::opt<bool, true> vcolumns("vcolumns",
    cl::desc("print character (column) numbers in diagnostics"),
    cl::location(global.params.showColumns));

cl::opt<bool, true> vgc("vgc",
    cl::desc("list all gc allocations including hidden ones"),
    cl::location(global.params.vgc));

cl::opt<bool, true, FlagParser> color("color",
    cl::desc("Force colored console output"),
    cl::location(global.params.color));

static cl::extrahelp footer("\n"
"-d-debug can also be specified without options, in which case it enables all\n"
"debug checks (i.e. (asserts, boundchecks, contracts and invariants) as well\n"
"as acting as -d-debug=1\n\n"
"Options marked with (*) also have a -disable-FOO variant with inverted\n"
"meaning.\n");

} // namespace opts
