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

#include "driver/exe_path.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/raw_ostream.h"
#if _WIN32
#include "Windows.h"
#else
#include <sys/stat.h>
#endif
#include <cassert>
#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#ifdef HAVE_SC_ARG_MAX
#include <unistd.h>
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
#if defined(_MSC_VER)
  _snprintf_s(bStr, _countof(bStr), sizeof(bStr), "%d", b);
#else
  snprintf(bStr, sizeof(bStr), "%d", b);
#endif
  return concat(a, bStr);
}

/**
 * Runs the given executable, returning its error code.
 */
int execute(const std::string &exePath, const char **args) {
#if LDC_LLVM_VER >= 700
  std::vector<llvm::StringRef> argv;
  for (auto arg = args; arg != nullptr; ++arg) {
    argv.push_back(*arg);
  }
  auto envVars = llvm::None;
#else
  auto argv = args;
  auto envVars = nullptr;
#endif

  std::string errorMsg;
  int rc = ls::ExecuteAndWait(exePath, argv, envVars,
#if LDC_LLVM_VER >= 600
                              {},
#else
                              nullptr,
#endif
                              0, 0, &errorMsg);
  if (!errorMsg.empty()) {
    error("Error executing %s: %s", exePath.c_str(), errorMsg.c_str());
  }
  return rc;
}

/**
 * Prints usage information to stdout.
 */
void printUsage(const char *argv0, const std::string &ldcPath) {
  // Print version information by actually invoking ldc -version.
  const char *args[] = {ldcPath.c_str(), "-version", nullptr};
  execute(ldcPath, args);

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
  -allinst         generate code for all template instantiations\n\
  -betterC         omit generating some runtime information and helper functions\n\
  -boundscheck=[on|safeonly|off]   bounds checks on, in @safe only, or off\n\
  -c               do not link\n\
  -color           turn colored console output on\n\
  -color=[on|off]  force colored console output on or off\n\
  -conf=<filename> use config file at filename\n\
  -cov             do code coverage analysis\n\
  -cov=<nnn>       require at least nnn%% code coverage\n\
  -D               generate documentation\n\
  -Dd<directory>   write documentation file to directory\n\
  -Df<filename>    write documentation file to filename\n\
  -d               silently allow deprecated features\n\
  -dw              show use of deprecated features as warnings (default)\n\
  -de              show use of deprecated features as errors (halt compilation)\n\
  -debug           compile in debug code\n\
  -debug=<level>   compile in debug code <= level\n\
  -debug=<ident>   compile in debug code identified by ident\n\
  -debuglib=<name> set symbolic debug library to name\n\
  -defaultlib=<name>\n\
                   set default library to name\n\
  -deps            print module dependencies (imports/file/version/debug/lib)\n\
  -deps=<filename> write module dependencies to filename (only imports)\n\
  -fPIC            generate position independent code\n\
  -dip25           implement http://wiki.dlang.org/DIP25 (experimental)\n\
  -dip1000         implement http://wiki.dlang.org/DIP1000 (experimental)\n\
  -dip1008         implement DIP1008 (experimental)\n\
  -g               add symbolic debug info\n\
  -gf              emit debug info for all referenced types\n\
  -gs              always emit stack frame\n"
#if 0
"  -gx              add stack stomp code\n"
#endif
"  -H               generate 'header' file\n\
  -Hd=<directory>  write 'header' file to directory\n\
  -Hf=<filename>   write 'header' file to filename\n\
  --help           print help and exit\n\
  -I=<directory>   look for imports also in directory\n\
  -i[=<pattern>]   include imported modules in the compilation\n\
  -ignore          ignore unsupported pragmas\n\
  -inline          do function inlining\n\
  -J=<directory>   look for string imports also in directory\n\
  -L=<linkerflag>  pass linkerflag to link\n\
  -lib             generate library rather than object files\n\
  -m32             generate 32 bit code\n"
#if 0
"  -m32mscoff       generate 32 bit code and write MS-COFF object files\n"
#endif
"  -m64             generate 64 bit code\n\
  -main            add default main() (e.g. for unittesting)\n\
  -man             open web browser on manual page\n"
#if 0
"  -map             generate linker .map file\n"
#endif
"  -mcpu=<id>       generate instructions for architecture identified by 'id'\n\
  -mcpu=?          list all architecture options\n\
  -mscrtlib=<name> MS C runtime library to reference from main/WinMain/DllMain\n\
  -mv=<package.module>=<filespec>  use <filespec> as source file for <package.module>\n\
  -noboundscheck   no array bounds checking (deprecated, use -boundscheck=off)\n\
  -O               optimize\n\
  -o-              do not write object file\n\
  -od=<directory>  write object & library files to directory\n\
  -of=<filename>   name output file to filename\n\
  -op              preserve source path for output files\n"
#if 0
"  -profile         profile runtime performance of generated code\n\
  -profile=gc      profile runtime allocations\n"
#endif
"  -release         compile release version\n\
  -shared          generate shared library (DLL)\n\
  -transition=<id> help with language change identified by 'id'\n\
  -transition=?    list all language changes\n\
  -unittest        compile in unit tests\n\
  -v               verbose\n\
  -vcolumns        print character (column) numbers in diagnostics\n\
  -vdmd            print the command used to invoke the underlying compiler\n\
  -verrors=<num>   limit the number of error messages (0 means unlimited)\n\
  -verrors=spec    show errors from speculative compiles such as __traits(compiles,...)\n\
  -vgc             list all gc allocations including hidden ones\n\
  -vtls            list all variables going into thread local storage\n\
  --version        print compiler version and exit\n\
  -version=<level> compile in version code >= level\n\
  -version=<ident> compile in version code identified by ident\n\
  -w               warnings as errors (compilation will halt)\n\
  -wi              warnings as messages (compilation will continue)\n\
  -X               generate JSON file\n\
  -Xf=<filename>   write JSON file to filename\n\n",
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
  char *env = getenv(envVarName);
  if (!env) {
    return;
  }

  env = strdup(env); // create our own writable copy

  size_t j = 1; // leave argv[0] alone
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
      j++;
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
void translateArgs(size_t originalArgc, char **originalArgv,
                   std::vector<const char *> &ldcArgs) {
  // Expand any response files present into the list of arguments.
  size_t argc = originalArgc;
  char **argv = originalArgv;
  if (response_expand(&argc, &argv)) {
    error("Could not read response file.");
  }

  std::vector<char *> args(argv, argv + argc);

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
  bool noFiles = true;
  bool pic = false; // -fPIC already encountered?

  for (size_t i = 1; i < args.size(); i++) {
    char *p = args[i];
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
      else if (strncmp(p + 1, "color", 5) == 0) {
        bool color = true;
        // Parse:
        //      -color
        //      -color=on|off
        if (p[6] == '=') {
          if (strcmp(p + 7, "off") == 0) {
            color = false;
          } else if (strcmp(p + 7, "on") != 0) {
            goto Lerror;
          }
        } else if (p[6]) {
          goto Lerror;
        }
        ldcArgs.push_back(color ? "-enable-color" : "-disable-color");
      }
      /* -conf
       * -cov
       * -shared
       */
      else if (strcmp(p + 1, "dylib") == 0) {
        ldcArgs.push_back("-shared");
      } else if (strcmp(p + 1, "fPIC") == 0) {
        if (!pic) {
          ldcArgs.push_back("-relocation-model=pic");
          pic = true;
        }
      } else if (strcmp(p + 1, "map") == 0) {
        goto Lnot_in_ldc;
      } else if (strcmp(p + 1, "multiobj") == 0) {
        goto Lnot_in_ldc;
      }
      /* -g
       * -gc
       */
      else if (strcmp(p + 1, "gf") == 0) {
        ldcArgs.push_back("-g");
      } else if (strcmp(p + 1, "gs") == 0) {
        ldcArgs.push_back("-disable-fp-elim");
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
      /* -mscrtlib
       */
      else if (strncmp(p + 1, "profile", 7) == 0) {
        if (p[8] == 0) {
          ldcArgs.push_back("-fdmd-trace-functions");
        } else if (strcmp(p + 8, "=gc") == 0) {
          goto Lnot_in_ldc; // ldcArgs.push_back("-fdmd-trace-gc");
        } else {
          goto Lerror;
        }
      }
      /* -v
       */
      else if (strcmp(p + 1, "vtls") == 0) {
        ldcArgs.push_back("-transition=tls");
      }
      /* -vcolumns
       * -vgc
       */
      else if (strncmp(p + 1, "verrors", 7) == 0) {
        if (p[8] == '=' && isdigit(static_cast<unsigned char>(p[9]))) {
          ldcArgs.push_back(p);
        } else if (strncmp(p + 9, "spec", 4) == 0) {
          ldcArgs.push_back("-verrors-spec");
        } else {
          goto Lerror;
        }
      } else if (strcmp(p + 1, "mcpu=?") == 0) {
        const char *mcpuargs[] = {ldcPath.c_str(), "-mcpu=help", nullptr};
        execute(ldcPath, mcpuargs);
        exit(EXIT_SUCCESS);
      } else if (strncmp(p + 1, "mcpu=", 5) == 0) {
        if (strcmp(p + 6, "baseline") == 0) {
          // ignore
        } else if (strcmp(p + 6, "avx") == 0) {
          ldcArgs.push_back("-mattr=+avx");
        } else if (strcmp(p + 6, "avx2") == 0) {
          ldcArgs.push_back("-mattr=+avx2");
        } else if (strcmp(p + 6, "native") == 0) {
          ldcArgs.push_back(p);
        } else {
          goto Lerror;
        }
      } else if (strcmp(p + 1, "transition=?") == 0) {
        const char *transitionargs[] = {ldcPath.c_str(), p, nullptr};
        execute(ldcPath, transitionargs);
        exit(EXIT_SUCCESS);
      }
      /* -transition=<id>
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
       * -unittest
       * -I
       * -J
       */
      else if (strncmp(p + 1, "debug", 5) == 0 && p[6] != 'l') {
        // Parse:
        //      -debug
        //      -debug=number
        //      -debug=identifier
        if (p[6] == '=') {
          if (isdigit(static_cast<unsigned char>(p[7]))) {
            long level;
            errno = 0;
            level = strtol(p + 7, &p, 10);
            if (*p || errno || level > INT_MAX) {
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
      } else if (strncmp(p + 1, "version", 7) == 0) {
        // Parse:
        //      -version=number
        //      -version=identifier
        if (p[8] == '=') {
          if (isdigit(static_cast<unsigned char>(p[9]))) {
            long level;
            errno = 0;
            level = strtol(p + 9, &p, 10);
            if (*p || errno || level > INT_MAX) {
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
        printUsage(originalArgv[0], ldcPath);
        exit(EXIT_SUCCESS);
      } else if (strcmp(p + 1, "-version") == 0) {
        // Print version information by actually invoking ldc -version.
        const char *versionargs[] = {ldcPath.c_str(), "-version", nullptr};
        execute(ldcPath, versionargs);
        exit(EXIT_SUCCESS);
      }
      /* -L
       * -defaultlib
       * -debuglib
       * -deps
       * -main
       */
      else if (strncmp(p + 1, "man", 3) == 0) {
        browse("http://wiki.dlang.org/LDC");
        exit(EXIT_SUCCESS);
      } else if (strcmp(p + 1, "run") == 0) {
        ldcArgs.insert(ldcArgs.end(), args.begin() + i, args.end());
        noFiles = (i == args.size() - 1);
        break;
      } else if (p[1] == '\0') {
        ldcArgs.push_back("-");
        noFiles = false;
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
      if (ext.equals_lower(".exe")) {
        // should be for Windows targets only
        ldcArgs.push_back(concat("-of=", p));
        continue;
      }
#ifdef _WIN32
      else if (strcmp(p, "/?") == 0) {
        printUsage(originalArgv[0], ldcPath);
        exit(EXIT_SUCCESS);
      }
#endif
      ldcArgs.push_back(p);
      noFiles = false;
    }
  }

  // at least one file is mandatory, except when `-Xi=…` is used
  if (noFiles && std::find_if(args.begin(), args.end(), [](const char *arg) {
                   return strncmp(arg, "-Xi=", 4) == 0;
                 }) == args.end()) {
    printUsage(originalArgv[0], ldcPath);
    if (originalArgc == 1)
      exit(EXIT_FAILURE); // compatible with DMD
    else
      error("No source file specified.");
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
 * Returns the OS-dependent length limit for the command line when invoking
 * subprocesses.
 */
size_t maxCommandLineLen() {
#if defined(HAVE_SC_ARG_MAX)
  // http://www.in-ulm.de/~mascheck/various/argmax – the factor 2 is just
  // a wild guess to account for the enviroment.
  return sysconf(_SC_ARG_MAX) / 2;
#elif defined(_WIN32)
  // http://blogs.msdn.com/b/oldnewthing/archive/2003/12/10/56028.aspx
  return 32767;
#else
#error "Do not know how to determine maximum command line length."
#endif
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

// In driver/main.d
int main(int argc, char **argv);

int cppmain(int argc, char **argv) {
  exe_path::initialize(argv[0]);

  std::string ldcExeName = LDC_EXE_NAME;
#ifdef _WIN32
  ldcExeName += ".exe";
#endif
  const std::string ldcPath = locateBinary(ldcExeName);
  if (ldcPath.empty()) {
    error("Could not locate " LDC_EXE_NAME " executable.");
  }

  // We need to manually set up argv[0] and the terminating NULL.
  std::vector<const char *> args;
  args.push_back(ldcPath.c_str());

  translateArgs(argc, argv, args);

  args.push_back(nullptr);

  // Check if we can get away without a response file.
  const size_t totalLen = std::accumulate(
      args.begin(), args.end() - 1,
      args.size() * 3, // quotes + space
      [](size_t acc, const char *arg) { return acc + strlen(arg); });
  if (totalLen <= maxCommandLineLen()) {
    return execute(ldcPath, args.data());
  }

  int rspFd;
  llvm::SmallString<128> rspPath;
  if (ls::fs::createUniqueFile("ldmd-%%-%%-%%-%%.rsp", rspFd, rspPath)) {
    error("Could not open temporary response file.");
  }

  {
    llvm::raw_fd_ostream rspOut(rspFd, /*shouldClose=*/true);
    // skip argv[0] and terminating NULL
    for (auto it = args.begin() + 1, end = args.end() - 1; it != end; ++it) {
      rspOut << *it << '\n';
    }
  }

  std::string rspArg = "@";
  rspArg += rspPath.str();

  args.resize(1);
  args.push_back(rspArg.c_str());
  args.push_back(nullptr);

  int rc = execute(ldcPath, args.data());

  if (ls::fs::remove(rspPath.str())) {
    warning("Could not remove response file.");
  }

  return rc;
}
