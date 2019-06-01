//===-- configfile.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/configfile.h"

#include "driver/args.h"
#include "driver/exe_path.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

#if _WIN32
#define WIN32_LEAN_AND_MEAN
#include "llvm/Support/ConvertUTF.h"
#include <windows.h>
#include <shlobj.h>
#include <tchar.h>
// Prevent name clash with LLVM
#undef GetCurrentDirectory
#endif

namespace sys = llvm::sys;

// dummy only; needs to be parsed manually earlier as the switches contained in
// the config file are injected into the command line options fed to the parser
static llvm::cl::opt<std::string>
    clConf("conf", llvm::cl::desc("Use configuration file <filename>"),
           llvm::cl::value_desc("filename"), llvm::cl::ZeroOrMore);

#if _WIN32
std::string getUserHomeDirectory() {
  char buff[MAX_PATH];
  HRESULT res = SHGetFolderPathA(NULL, CSIDL_FLAG_CREATE | CSIDL_APPDATA, NULL,
                                 SHGFP_TYPE_CURRENT, buff);
  if (res != S_OK)
    assert(0 && "Failed to get user home directory");
  return buff;
}
#else
std::string getUserHomeDirectory() {
  const char *home = getenv("HOME");
  return home ? home : "/";
}
#endif

#if _WIN32
static bool ReadPathFromRegistry(llvm::SmallString<128> &p) {
  HKEY hkey;
  bool res = false;
  // FIXME: Version number should be a define.
  if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                   _T("SOFTWARE\\ldc-developers\\LDC\\0.11.0"), NULL,
                   KEY_QUERY_VALUE, &hkey) == ERROR_SUCCESS) {
    DWORD length;
    if (RegGetValue(hkey, NULL, _T("Path"), RRF_RT_REG_SZ, NULL, NULL,
                    &length) == ERROR_SUCCESS) {
      std::vector<TCHAR> buffer;
      buffer.reserve(length);
      const auto data = buffer.data();
      if (RegGetValue(hkey, NULL, _T("Path"), RRF_RT_REG_SZ, NULL, data,
                      &length) == ERROR_SUCCESS) {
#if UNICODE
#if LDC_LLVM_VER >= 400
        using UTF16 = llvm::UTF16;
#endif
        std::string out;
        res = llvm::convertUTF16ToUTF8String(
            llvm::ArrayRef<UTF16>(reinterpret_cast<UTF16 *>(data), length),
            out);
        p = out;
#else
        p = std::string(data);
        res = true;
#endif
      }
    }
    RegCloseKey(hkey);
  }
  return res;
}
#endif

bool ConfigFile::locate(std::string &pathstr) {
  // temporary configuration

  llvm::SmallString<128> p;
  const char *filename = "ldc2.conf";

#define APPEND_FILENAME_AND_RETURN_IF_EXISTS                                   \
  {                                                                            \
    sys::path::append(p, filename);                                            \
    if (sys::fs::exists(p.str())) {                                            \
      pathstr = p.str();                                                       \
      return true;                                                             \
    }                                                                          \
  }

  // try the current working dir
  if (!sys::fs::current_path(p))
    APPEND_FILENAME_AND_RETURN_IF_EXISTS

  // try next to the executable
  p = exe_path::getBinDir();
  APPEND_FILENAME_AND_RETURN_IF_EXISTS

  // user configuration

  // try ~/.ldc
  p = getUserHomeDirectory();
  sys::path::append(p, ".ldc");
  APPEND_FILENAME_AND_RETURN_IF_EXISTS

#if _WIN32
  // try home dir
  p = getUserHomeDirectory();
  APPEND_FILENAME_AND_RETURN_IF_EXISTS
#endif

  // system configuration

  // try in etc relative to the executable: exe\..\etc
  // do not use .. in path because of security risks
  p = exe_path::getBaseDir();
  if (!p.empty()) {
    sys::path::append(p, "etc");
    APPEND_FILENAME_AND_RETURN_IF_EXISTS
  }

#if _WIN32
  // Try reading path from registry
  if (ReadPathFromRegistry(p)) {
    sys::path::append(p, "etc");
    APPEND_FILENAME_AND_RETURN_IF_EXISTS
  }
#else
#define STR(x) #x
#define XSTR(x) STR(x)
  // try the install-prefix/etc
  p = XSTR(LDC_INSTALL_PREFIX);
  sys::path::append(p, "etc");
  APPEND_FILENAME_AND_RETURN_IF_EXISTS

  // try the install-prefix/etc/ldc
  p = XSTR(LDC_INSTALL_PREFIX);
  sys::path::append(p, "etc");
  sys::path::append(p, "ldc");
  APPEND_FILENAME_AND_RETURN_IF_EXISTS
#undef XSTR
#undef STR

  // try /etc (absolute path)
  p = "/etc";
  APPEND_FILENAME_AND_RETURN_IF_EXISTS

  // try /etc/ldc (absolute path)
  p = "/etc/ldc";
  APPEND_FILENAME_AND_RETURN_IF_EXISTS
#endif

#undef APPEND_FILENAME_AND_RETURN_IF_EXISTS

  fprintf(stderr, "Warning: failed to locate the configuration file %s\n",
          filename);
  return false;
}

bool ConfigFile::read(const char *explicitConfFile, const char *triple) {
  std::string pathstr;
  // explicitly provided by user in command line?
  if (explicitConfFile) {
    const std::string clPath = explicitConfFile;
    // an empty path (`-conf=`) means no config file
    if (clPath.empty())
      return true;

    if (sys::fs::exists(clPath)) {
      pathstr = clPath;
    } else {
      fprintf(stderr,
              "Warning: configuration file '%s' not found, falling "
              "back to default\n",
              clPath.c_str());
    }
  }

  // locate file automatically if path is not set yet
  if (pathstr.empty()) {
    if (!locate(pathstr)) {
      return false;
    }
  }

  pathcstr = strdup(pathstr.c_str());
  auto binpath = exe_path::getBinDir();

  return readConfig(pathcstr, triple, binpath.c_str());
}

void ConfigFile::extendCommandLine(llvm::SmallVectorImpl<const char *> &args) {
  // insert 'switches' before all user switches
  args.insert(args.begin() + 1, switches.begin(), switches.end());

  // append 'post-switches', but before a first potential '-run'
  size_t runIndex = 0;
  for (size_t i = 1; i < args.size(); ++i) {
    if (args::isRunArg(args[i])) {
      runIndex = i;
      break;
    }
  }
  args.insert(runIndex == 0 ? args.end() : args.begin() + runIndex,
              postSwitches.begin(), postSwitches.end());
}

bool ConfigFile::sectionMatches(const char *section, const char *triple) {
  return llvm::Regex(section).match(triple);
}
