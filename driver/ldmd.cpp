//===-- ldmd.cpp - Drop-in DMD replacement wrapper for LDC ----------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license, except for the
// command line handling code, which originated from DMD. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Wrapper allowing use of LDC as drop-in replacement for DMD.
//
// Most command-line options are passed through to LDC; some with different
// names or semantics need to be translated.
//
// DMD also reads switches from the DFLAGS enviroment variable, if present. This
// is contrary to what C compilers do, where CFLAGS is usually handled by the
// build system.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_EXE_NAME
#error "Please define LDC_EXE_NAME to the name of the LDC executable to use."
#endif

#include "driver/args.h"
#include "driver/exe_path.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace ls = llvm::sys;

// We reuse DMD's response file parsing routine for maximum compatibility - it
// handles quotes in a very peculiar way.
int response_expand(size_t *pargc, char ***pargv);

// in dmd/root/man.d
void browse(const char *url);

/**
 * Prints a formatted error message to stderr and exits the program.
 */
void error(const char *fmt, ...) {
  va_list argp;
  va_start(argp, fmt);
  fprintf(stderr, "Error: ");
  vfprintf(stderr, fmt, argp);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
  va_end(argp);
}

/**
 * Prints a formatted warning message to stderr.
 */
void warning(const char *fmt, ...) {
  va_list argp;
  va_start(argp, fmt);
  fprintf(stderr, "Warning: ");
  vfprintf(stderr, fmt, argp);
  fprintf(stderr, "\n");
  va_end(argp);
}

char *concat(const char *a, const char *b) {
  size_t na = strlen(a);
  size_t nb = strlen(b);
  char *result = static_cast<char *>(malloc(na + nb + 1));
  assert(result);
  memcpy(result, a, na);
  memcpy(result + na, b, nb + 1);
  return result;
}

char *concat(const char *a, int b) {
  char bStr[14];
  snprintf(bStr, sizeof(bStr), "%d", b);
  return concat(a, bStr);
}

template <int N>
bool startsWith(const char *str, const char (&prefix)[N]) {
  // N includes terminating null
  return strncmp(str, prefix, N - 1) == 0;
}

/**
 * Runs the given executable, returning its error code.
 */
int execute(std::vector<const char *> fullArgs) {
  std::string errorMsg;
  const char *executable = fullArgs[0];
  const int rc =
      args::executeAndWait(std::move(fullArgs), llvm::sys::WEM_UTF8, &errorMsg);
  if (rc && !errorMsg.empty()) {
    error("Error executing %s: %s", executable, errorMsg.c_str());
  }
  return rc;
}

/**
 * Prints usage information to stdout.
 */
void printUsage(const char *argv0, const std::string &ldcPath) {
  // Print version information by actually invoking ldc -version.
  execute({ldcPath.c_str(), "-version"});

  printf(
      "\n\
Usage:\n\
  %s [<option>...] <file>...\n\
  %s [<option>...] -run <file> [<arg>...]\n\
\n\
Where:\n\
  <file>           D source file\n\
  <arg>            Argument to pass when running the resulting program\n\
\n\
<option>:\n\
  @<cmdfile>       read arguments from cmdfile\n\
  -allinst          generate code for all template instantiations\n\
  -betterC          omit generating some runtime information and helper functions\n\
  -boundscheck=[on|safeonly|off]\n\
                    bounds checks on, in @safe only, or off\n\
  -c                compile only, do not link\n\
  -check=[assert|bounds|in|invariant|out|switch][=[on|off]]\n\
                    enable or disable specific checks\n"
#if 0
"  -check=[h|help|?] list information on all available checks\n"
#endif
"  -checkaction=[D|C|halt|context]\n\
                    behavior on assert/boundscheck/finalswitch failure\n"
#if 0
"  -checkaction=[h|help|?]\n\
                    list information on all available check actions\n"
#endif
"  -color            turn colored console output on\n\
  -color=[on|off|auto]\n\
                    force colored console output on or off, or only when not redirected (default)\n\
  -conf=<filename>  use config file at filename\n\
  -cov              do code coverage analysis\n\
  -cov=ctfe         include code executed during CTFE in coverage report\n\
  -cov=<nnn>        require at least nnn%% code coverage\n"
#if 0
"  -cpp=<filename>   use filename as the name of the C preprocessor to use for ImportC files\n"
#endif
"  -D                generate documentation\n\
  -Dd<directory>    write documentation file to directory\n\
  -Df<filename>     write documentation file to filename\n\
  -d                silently allow deprecated features and symbols\n\
  -de               issue an error when deprecated features or symbols are used (halt compilation)\n\
  -dw               issue a message when deprecated features or symbols are used (default)\n\
  -debug            compile in debug code\n\
  -debug=<level>    compile in debug code <= level\n\
  -debug=<ident>    compile in debug code identified by ident\n\
  -debuglib=<name>  set symbolic debug library to name\n\
  -defaultlib=<name>\n\
                    set default library to name\n\
  -deps             print module dependencies (imports/file/version/debug/lib)\n\
  -deps=<filename>  write module dependencies to filename (only imports)\n\
  -dllimport=<value>\n\
                    Windows only: select symbols to dllimport (none/defaultLibsOnly/all)\n\
  -extern-std=<standard>\n\
                    set C++ name mangling compatibility with <standard>\n"
#if 0
"  -extern-std=[h|help|?]\n\
                    list all supported standards\n"
#endif
"  -fIBT             generate Indirect Branch Tracking code\n\
  -fPIC             generate position independent code\n"
#if 0
"  -fPIE             generate position independent executables\n"
#endif
"  -g                add symbolic debug info\n\
  -gdwarf=<version> add DWARF symbolic debug info\n\
  -gf               emit debug info for all referenced types\n\
  -gs               always emit stack frame\n"
#if 0
"  -gx               add stack stomp code\n"
#endif
"  -H                generate 'header' file\n\
  -Hd=<directory>   write 'header' file to directory\n\
  -Hf=<filename>    write 'header' file to filename\n\
  -HC[=[silent|verbose]]\n\
                    generate C++ 'header' file\n"
#if 0
"  -HC=[?|h|help]    list available modes for C++ 'header' file generation\n"
#endif
"  -HCd=<directory>  write C++ 'header' file to directory\n\
  -HCf=<filename>   write C++ 'header' file to filename\n\
  --help            print help and exit\n\
  -I=<directory>    look for imports also in directory\n\
  -i[=<pattern>]    include imported modules in the compilation\n\
  -ignore           deprecated flag, unsupported pragmas are always ignored now\n\
  -inline           do function inlining\n\
  -J=<directory>    look for string imports also in directory\n\
  -L=<linkerflag>   pass linkerflag to link\n\
  -lib              generate library rather than object files\n\
  -lowmem           enable garbage collection for the compiler\n\
  -m32              generate 32 bit code\n"
#if 0
"  -m32mscoff        generate 32 bit code and write MS-COFF object files\n"
#endif
"  -m64              generate 64 bit code\n\
  -main             add default main() if not present already (e.g. for unittesting)\n\
  -makedeps[=<filename>]\n\
                    print dependencies in Makefile compatible format to filename or stdout\n\
  -man              open web browser on manual page\n"
#if 0
"  -map              generate linker .map file\n"
#endif
"  -mcpu=<id>        generate instructions for architecture identified by 'id'\n\
  -mcpu=[h|help|?]  list all architecture options\n\
  -mixin=<filename> expand and save mixins to file specified by <filename>\n\
  -mscrtlib=<libname>\n\
                    MS C runtime library to reference from main/WinMain/DllMain\n\
  -mv=<package.module>=<filespec>\n\
                    use <filespec> as source file for <package.module>\n\
  -noboundscheck    no array bounds checking (deprecated, use -boundscheck=off)\n\
  -nothrow          assume no Exceptions will be thrown\n\
  -O                optimize\n\
  -o-               do not write object file\n\
  -od=<directory>   write object & library files to directory\n\
  -of=<filename>    name output file to filename\n\
  -op               preserve source path for output files\n"
#if 0
"  -os=<os>          sets target operating system to <os>\n"
#endif
"  -P=<preprocessorflag>\n\
                    pass preprocessorflag to C preprocessor\n\
  -preview=<name>   enable an upcoming language change identified by 'name'\n\
  -preview=[h|help|?]\n\
                    list all upcoming language changes\n\
  -profile          profile runtime performance of generated code\n"
#if 0
"  -profile=gc       profile runtime allocations\n"
#endif
"  -release          contracts and asserts are not emitted, and bounds checking is performed only in @safe functions\n\
  -revert=<name>    revert language change identified by 'name'\n\
  -revert=[h|help|?]\n\
                    list all revertable language changes\n\
  -run <srcfile>    compile, link, and run the program srcfile\n\
  -shared           generate shared library (DLL)\n\
  -target=<triple>  use <triple> as <arch>-[<vendor>-]<os>[-<cenv>[-<cppenv]]\n\
  -transition=<name>\n\
                    help with language change identified by 'name'\n\
  -transition=[h|help|?]\n\
                    list all language changes\n\
  -unittest         compile in unit tests\n\
  -v                verbose\n\
  -vasm             generate additional textual assembly files (*.s)\n\
  -vcolumns         print character (column) numbers in diagnostics\n\
  -vdmd             print the underlying LDC command line\n\
  -verror-style=[digitalmars|gnu]\n\
                    set the style for file/line number annotations on compiler messages\n\
  -verror-supplements=<num>\n\
                    limit the number of supplemental messages for each error (0 means unlimited)\n\
  -verrors=<num>    limit the number of error messages (0 means unlimited)\n\
  -verrors=context  show error messages with the context of the erroring source line\n\
  -verrors=spec     show errors from speculative compiles such as __traits(compiles,...)\n\
  --version         print compiler version and exit\n\
  -version=<level>  compile in version code >= level\n\
  -version=<ident>  compile in version code identified by ident\n\
  -vgc              list all gc allocations including hidden ones\n\
  -visibility=<value>\n\
                    default visibility of symbols (default/hidden/public)\n\
  -vtemplates=[list-instances]\n\
                    list statistics on template instantiations\n\
  -vtls             list all variables going into thread local storage\n\
  -w                warnings as errors (compilation will halt)\n\
  -wi               warnings as messages (compilation will continue)\n\
  -X                generate JSON file\n\
  -Xf=<filename>    write JSON file to filename\n\
  -Xcc=<driverflag> pass driverflag to linker driver (cc)\n",
      argv0, argv0);
}

/**
 * Parses an enviroment variable for flags and appends them to given list of
 * arguments.
 *
 * This corresponds to getenv_setargv() in DMD, but we need to duplicate it
 * here since it is defined in mars.c.
 */
void appendEnvVar(const char *envVarName, std::vector<char *> &args) {
  std::string envVar = env::get(envVarName);
  if (envVar.empty()) {
    return;
  }

  char *env = strdup(envVar.c_str()); // create forever-living copy

  while (1) {
    switch (*env) {
    case ' ':
    case '\t':
      env++;
      break;

    case 0:
      return;

    default:
      args.push_back(env); // append
      char *p = env;
      int slash = 0;
      int instring = 0;
      char c = 0;

      while (1) {
        c = *env++;
        switch (c) {
        case '"':
          p -= (slash >> 1);
          if (slash & 1) {
            p--;
            goto Laddc;
          }
          instring ^= 1;
          slash = 0;
          continue;

        case ' ':
        case '\t':
          if (instring) {
            goto Laddc;
          }
          *p = 0;
          break;

        case '\\':
          slash++;
          *p++ = c;
          continue;

        case 0:
          *p = 0;
          return;

        default:
        Laddc:
          slash = 0;
          *p++ = c;
          continue;
        }
        break;
      }
    }
  }
}

/**
 * Translates the LDMD command-line args (incl. DFLAGS environment variable)
 * to LDC args.
 * `ldcArgs` needs to be initialized with the path to the LDC executable.
 */
void translateArgs(const llvm::SmallVectorImpl<const char *> &ldmdArgs,
                   std::vector<const char *> &ldcArgs) {
  // Expand any response files present into the list of arguments.
  size_t argc = ldmdArgs.size();
  char **argv = const_cast<char **>(ldmdArgs.data());
  if (response_expand(&argc, &argv)) {
    error("Could not read response file.");
  }

  std::vector<const char *> args(argv, argv + argc);

  std::vector<char *> dflags;
  appendEnvVar("DFLAGS", dflags);
  if (!dflags.empty()) {
    // append, but before a first potential '-run'
    size_t runIndex = 0;
    for (size_t i = 1; i < args.size(); ++i) {
      if (strcmp(args[i], "-run") == 0) {
        runIndex = i;
        break;
      }
    }
    args.insert(runIndex == 0 ? args.end() : args.begin() + runIndex,
                dflags.begin(), dflags.end());
  }

  assert(ldcArgs.size() == 1);
  const std::string ldcPath = ldcArgs[0];

  ldcArgs.push_back("-ldmd");

  bool vdmd = false;
  bool pic = false; // -fPIC already encountered?

  for (size_t i = 1; i < args.size(); i++) {
    const char *p = args[i];
    if (*p == '-') {
      if (strcmp(p + 1, "vdmd") == 0) {
        vdmd = true;
      }
      /* Most args are handled directly by LDC.
       * Order corresponds to parsing order in dmd's mars.d.
       *
       * -allinst
       * -de
       * -d
       * -dw
       * -c
       */
      else if (startsWith(p + 1, "check=")) {
        // Parse:
        //      -check=[assert|bounds|in|invariant|out|switch][=[on|off]]
        const char *arg = p + 7;
        if (strcmp(arg, "on") == 0) {
          ldcArgs.push_back("-boundscheck=on");
          ldcArgs.push_back("-enable-asserts");
          ldcArgs.push_back("-enable-preconditions");
          ldcArgs.push_back("-enable-invariants");
          ldcArgs.push_back("-enable-postconditions");
          ldcArgs.push_back("-enable-switch-errors");
        } else if (strcmp(arg, "off") == 0) {
          ldcArgs.push_back("-boundscheck=off");
          ldcArgs.push_back("-disable-asserts");
          ldcArgs.push_back("-disable-preconditions");
          ldcArgs.push_back("-disable-invariants");
          ldcArgs.push_back("-disable-postconditions");
          ldcArgs.push_back("-disable-switch-errors");
        } else {
          const auto argLength = strlen(arg);
          bool enabled = false;
          size_t kindLength = 0;
          if (argLength > 3 && memcmp(arg + argLength - 3, "=on", 3) == 0) {
            enabled = true;
            kindLength = argLength - 3;
          } else if (argLength > 4 &&
                     memcmp(arg + argLength - 4, "=off", 4) == 0) {
            enabled = false;
            kindLength = argLength - 4;
          } else {
            enabled = true;
            kindLength = argLength;
          }

          const auto check = [&](size_t dmdLength, const char *dmd,
                                 const char *ldc) {
            if (kindLength == dmdLength && memcmp(arg, dmd, dmdLength) == 0) {
              ldcArgs.push_back(
                  concat(enabled ? "-enable-" : "-disable-", ldc));
              return true;
            }
            return false;
          };

          if (kindLength == 6 && memcmp(arg, "bounds", 6) == 0) {
            ldcArgs.push_back(enabled ? "-boundscheck=on" : "-boundscheck=off");
          } else if (!(check(6, "assert", "asserts") ||
                       check(2, "in", "preconditions") ||
                       check(9, "invariant", "invariants") ||
                       check(3, "out", "postconditions") ||
                       check(6, "switch", "switch-errors"))) {
            goto Lerror;
          }
        }
      }
      /* -checkaction
       */
      else if (startsWith(p + 1, "color")) {
        // Parse:
        //      -color
        //      -color=auto|on|off
        if (p[6] == '=') {
          if (strcmp(p + 7, "on") == 0) {
            ldcArgs.push_back("-enable-color");
          } else if (strcmp(p + 7, "off") == 0) {
            ldcArgs.push_back("-disable-color");
          } else if (strcmp(p + 7, "auto") != 0) {
            goto Lerror;
          }
        } else if (p[6]) {
          goto Lerror;
        } else {
          ldcArgs.push_back("-enable-color");
        }
      }
      /* -conf
       * -cov
       * -shared
       */
      else if (startsWith(p + 1, "visibility=")) {
        ldcArgs.push_back(concat("-fvisibility=", p + 12));
      }
      /* -dllimport
       */
      else if (strcmp(p + 1, "dylib") == 0) {
        ldcArgs.push_back("-shared");
      } else if (strcmp(p + 1, "fIBT") == 0) {
        ldcArgs.push_back("-fcf-protection=branch");
      } else if (strcmp(p + 1, "fPIC") == 0) {
        if (!pic) {
          ldcArgs.push_back("-relocation-model=pic");
          pic = true;
        }
      } else if (strcmp(p + 1, "fPIE") == 0) {
        goto Lnot_in_ldc;
      } else if (strcmp(p + 1, "map") == 0) {
        goto Lnot_in_ldc;
      } else if (strcmp(p + 1, "multiobj") == 0) {
        goto Lnot_in_ldc;
      }
      /* -g
       * -gc
       */
      else if (startsWith(p + 1, "gdwarf=")) {
        ldcArgs.push_back("-gdwarf"); // implies -g and enforces DWARF for MSVC
        ldcArgs.push_back("-dwarf-version");
        ldcArgs.push_back(p + 8);
      } else if (strcmp(p + 1, "gf") == 0) {
        ldcArgs.push_back("-g");
      } else if (strcmp(p + 1, "gs") == 0) {
        ldcArgs.push_back("-frame-pointer=all");
      } else if (strcmp(p + 1, "gx") == 0) {
        goto Lnot_in_ldc;
      } else if (strcmp(p + 1, "gt") == 0) {
        error("use -profile instead of -gt\n");
      }
      /* -m32
       * -m64
       */
      else if (strcmp(p + 1, "m32mscoff") == 0) {
        ldcArgs.push_back("-m32");
      }
      /* -mixin
       * -mscrtlib
       */
      else if (startsWith(p + 1, "profile")) {
        if (p[8] == 0) {
          ldcArgs.push_back("-fdmd-trace-functions");
        } else if (strcmp(p + 8, "=gc") == 0) {
          goto Lnot_in_ldc; // ldcArgs.push_back("-fdmd-trace-gc");
        } else {
          goto Lerror;
        }
      }
      /* -v
       * -vcg-ast
       */
      else if (strcmp(p + 1, "vasm") == 0) {
        ldcArgs.push_back("--output-s");
        ldcArgs.push_back("--output-o");
        ldcArgs.push_back("--x86-asm-syntax=intel");
      } else if (strcmp(p + 1, "vtls") == 0) {
        ldcArgs.push_back("-transition=tls");
      }
      /* -vtemplates
       * -vcolumns
       * -vgc
       */
      else if (startsWith(p + 1, "verrors")) {
        if (p[8] == '=' && isdigit(static_cast<unsigned char>(p[9]))) {
          ldcArgs.push_back(p);
        } else if (startsWith(p + 9, "spec")) {
          ldcArgs.push_back("-verrors-spec");
        } else if (startsWith(p + 9, "context")) {
          ldcArgs.push_back("-verrors-context");
        } else {
          goto Lerror;
        }
      }
      /* -verror-supplements
       * -verror-style
       */
      else if (startsWith(p + 1, "target=")) {
        ldcArgs.push_back(concat("-mtriple=", p + 8));
      } else if (startsWith(p + 1, "mcpu=")) {
        const char *c = p + 6;
        if (strcmp(c, "?") == 0 || strcmp(c, "h") == 0 ||
            strcmp(c, "help") == 0) {
          execute({ldcPath.c_str(), "-mcpu=help"});
          exit(EXIT_SUCCESS);
        } else if (strcmp(c, "baseline") == 0) {
          // ignore
        } else if (strcmp(c, "avx") == 0) {
          ldcArgs.push_back("-mattr=+avx");
        } else if (strcmp(c, "avx2") == 0) {
          ldcArgs.push_back("-mattr=+avx2");
        } else if (strcmp(c, "native") == 0) {
          ldcArgs.push_back(p);
        } else {
          goto Lerror;
        }
      } else if (startsWith(p + 1, "os=")) {
        error("please specify a full target triple via -mtriple instead of the "
              "target OS (-os) alone");
      }
      /* -extern-std
       * -transition
       * -preview
       * -revert
       * -w
       * -wi
       * -O
       * -o-
       * -od
       * -of
       * -op
       */
      else if (strcmp(p + 1, "o") == 0) {
        error("-o no longer supported, use -of or -od");
      }
      /* -D
       * -Dd
       * -Df
       * -HC
       * -HCd
       * -HCf
       * -H
       * -Hd
       * -Hf
       * -X
       * -Xf
       * -ignore
       * -property
       */
      else if (strcmp(p + 1, "inline") == 0) {
        ldcArgs.push_back("-enable-inlining");
        ldcArgs.push_back("-Hkeep-all-bodies");
      }
      /* -dip25
       * -dip1000
       * -dip1008
       */
      else if (strcmp(p + 1, "lib") == 0) {
        ldcArgs.push_back(p);
        // DMD seems to emit objects directly into the static lib being
        // generated. No object files are created and therefore they never
        // collide due to duplicate .d filenames (in different dirs).
        // Approximate that behavior by naming the object files uniquely via -oq
        // and instructing LDC to remove the object files on success.
        ldcArgs.push_back("-oq");
        ldcArgs.push_back("-cleanup-obj");
      } else if (strcmp(p + 1, "nofloat") == 0) {
        goto Lnot_in_ldc;
      } else if (strcmp(p + 1, "quiet") == 0) {
        // ignore
      }
      /* -release
       * -betterC
       */
      else if (strcmp(p + 1, "noboundscheck") == 0) {
        ldcArgs.push_back("-boundscheck=off");
      }
      /* -boundscheck
       */
      else if (strcmp(p + 1, "nothrow") == 0) {
        ldcArgs.push_back("-fno-exceptions");
      }
      /* -unittest
       * -I
       * -J
       */
      else if (startsWith(p + 1, "debug") && p[6] != 'l') {
        // Parse:
        //      -debug
        //      -debug=number
        //      -debug=identifier
        if (p[6] == '=') {
          if (isdigit(static_cast<unsigned char>(p[7]))) {
            long level;
            errno = 0;
            char *end;
            level = strtol(p + 7, &end, 10);
            if (*end || errno || level > INT_MAX) {
              goto Lerror;
            }
            ldcArgs.push_back(concat("-d-debug=", static_cast<int>(level)));
          } else {
            ldcArgs.push_back(concat("-d-debug=", p + 7));
          }
        } else if (p[6]) {
          goto Lerror;
        } else {
          ldcArgs.push_back("-d-debug");
        }
      } else if (startsWith(p + 1, "version")) {
        // Parse:
        //      -version=number
        //      -version=identifier
        if (p[8] == '=') {
          if (isdigit(static_cast<unsigned char>(p[9]))) {
            long level;
            errno = 0;
            char *end;
            level = strtol(p + 9, &end, 10);
            if (*end || errno || level > INT_MAX) {
              goto Lerror;
            }
            ldcArgs.push_back(concat("-d-version=", static_cast<int>(level)));
          } else {
            ldcArgs.push_back(concat("-d-version=", p + 9));
          }
        } else {
          goto Lerror;
        }
      } else if (strcmp(p + 1, "-b") == 0 || strcmp(p + 1, "-c") == 0 ||
                 strcmp(p + 1, "-f") == 0 || strcmp(p + 1, "-r") == 0 ||
                 strcmp(p + 1, "-x") == 0 || strcmp(p + 1, "-y") == 0) {
        ldcArgs.push_back(concat("-hidden-debug-", p + 2));
      } else if (strcmp(p + 1, "-help") == 0 || strcmp(p + 1, "h") == 0) {
        printUsage(ldmdArgs[0], ldcPath);
        exit(EXIT_SUCCESS);
      } else if (strcmp(p + 1, "-version") == 0) {
        // Print version information by actually invoking ldc -version.
        execute({ldcPath.c_str(), "-version"});
        exit(EXIT_SUCCESS);
      }
      /* -L
       * -P
       * -defaultlib
       * -debuglib
       * -deps
       * -main
       */
      else if (startsWith(p + 1, "man")) {
        browse("http://wiki.dlang.org/LDC");
        exit(EXIT_SUCCESS);
      } else if (strcmp(p + 1, "run") == 0) {
        ldcArgs.insert(ldcArgs.end(), args.begin() + i, args.end());
        break;
      } else if (p[1] == '\0') {
        ldcArgs.push_back("-");
      } else if (p[1] == 'C') {
        ldcArgs.push_back(concat("-", p + 2));
      } else {
      Lerror:
        ldcArgs.push_back(p);
        continue;

      Lnot_in_ldc:
        warning("command-line option '%s' not yet supported by LDC.", p);
        continue;
      }
    } else {
      const auto ext = ls::path::extension(p);
      if (
#if LDC_LLVM_VER >= 1300
        ext.equals_insensitive(".exe")
#else
        ext.equals_lower(".exe")
#endif
          ) {
        // should be for Windows targets only
        ldcArgs.push_back(concat("-of=", p));
        continue;
      }
#ifdef _WIN32
      else if (strcmp(p, "/?") == 0) {
        printUsage(ldmdArgs[0], ldcPath);
        exit(EXIT_SUCCESS);
      }
#endif
      ldcArgs.push_back(p);
    }
  }

  if (vdmd) {
    printf(" -- Invoking:");
    for (const auto &arg : ldcArgs) {
      printf(" %s", arg);
    }
    puts("");
  }
}

/**
 * Tries to locate an executable with the given name, or an invalid path if
 * nothing was found. Search paths: 1. Directory where this binary resides.
 * 2. System PATH.
 */
std::string locateBinary(std::string exeName) {
  std::string path = exe_path::prependBinDir(exeName.c_str());
  if (ls::fs::can_execute(path)) {
    return path;
  }

  llvm::ErrorOr<std::string> res = ls::findProgramByName(exeName);
  path = res ? res.get() : std::string();
  if (ls::fs::can_execute(path)) {
    return path;
  }

  return "";
}

static llvm::SmallVector<const char *, 32> ldmdArguments;

/// LDMD's entry point, C main.
#if LDC_WINDOWS_WMAIN
int wmain(int argc, const wchar_t **originalArgv)
#else
int main(int argc, const char **originalArgv)
#endif
{
  // Initialize `ldmdArguments` with the UTF-8 command-line args.
  args::getCommandLineArguments(argc, originalArgv, ldmdArguments);

  // Move on to _d_run_main, _Dmain, and finally cppmain below.
  // Only pass the first arg to skip useless work, e.g., not applying --DRT-* to
  // LDMD itself.
  return args::forwardToDruntime(1, originalArgv);
}

int cppmain() {
  exe_path::initialize(ldmdArguments[0]);

  std::string ldcExeName = LDC_EXE_NAME;
#ifdef _WIN32
  ldcExeName += ".exe";
#endif
  const std::string ldcPath = locateBinary(ldcExeName);
  if (ldcPath.empty()) {
    error("Could not locate " LDC_EXE_NAME " executable.");
  }

  if (ldmdArguments.size() == 1) {
    printUsage(ldmdArguments[0], ldcPath);
    exit(EXIT_FAILURE);
  }

  std::vector<const char *> fullArgs;
  fullArgs.push_back(ldcPath.c_str());

  translateArgs(ldmdArguments, fullArgs);

  return execute(std::move(fullArgs));
}
