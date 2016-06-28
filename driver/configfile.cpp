//===-- configfile.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/configfile.h"
#include "driver/exe_path.h"
#include "mars.h"
#include "libconfig.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
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
llvm::cl::opt<std::string>
    clConf("conf", llvm::cl::desc("Use configuration file <filename>"),
           llvm::cl::value_desc("filename"));

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
  if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, _T("SOFTWARE\\ldc-developers\\LDC\\0.11.0"),
                   NULL, KEY_QUERY_VALUE, &hkey) == ERROR_SUCCESS) {
    DWORD length;
    if (RegGetValue(hkey, NULL, _T("Path"), RRF_RT_REG_SZ, NULL, NULL, &length) ==
        ERROR_SUCCESS) {
      TCHAR *data = static_cast<TCHAR *>(_alloca(length * sizeof(TCHAR)));
      if (RegGetValue(hkey, NULL, _T("Path"), RRF_RT_REG_SZ, NULL, data, &length) ==
          ERROR_SUCCESS) {
#if UNICODE
        std::string out;
        res = llvm::convertUTF16ToUTF8String(
            llvm::ArrayRef<UTF16>(reinterpret_cast<UTF16 *>(data), length), out);
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

ConfigFile::ConfigFile() {
  cfg = new config_t;
  config_init(cfg);
}

bool ConfigFile::locate() {
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

bool ConfigFile::read(const char *explicitConfFile, const char* section) {
  // explicitly provided by user in command line?
  if (explicitConfFile) {
    const std::string clPath = explicitConfFile;
    // treat an empty path (`-conf=`) as missing command-line option,
    // defaulting to an auto-located config file, analogous to DMD
    if (!clPath.empty()) {
      if (sys::fs::exists(clPath)) {
        pathstr = clPath;
      } else {
        fprintf(stderr, "Warning: configuration file '%s' not found, falling "
                        "back to default\n",
                clPath.c_str());
      }
    }
  }

  // locate file automatically if path is not set yet
  if (pathstr.empty()) {
    if (!locate()) {
      return false;
    }
  }

  // read the cfg
  if (!config_read_file(cfg, pathstr.c_str())) {
    std::cerr << "error reading configuration file" << std::endl;
    return false;
  }

  config_setting_t *root = nullptr;
  if (section && *section)
    root = config_lookup(cfg, section);

  // make sure there's a default group
  if (!root) {
    section = "default";
    root = config_lookup(cfg, section);
  }
  if (!root) {
    std::cerr << "no default settings in configuration file" << std::endl;
    return false;
  }
  if (!config_setting_is_group(root)) {
    std::cerr << section << " is not a group" << std::endl;
    return false;
  }

  // handle switches
  if (config_setting_t *sw = config_setting_get_member(root, "switches")) {
    // replace all %%ldcbinarypath%% occurrences by the path to the
    // LDC bin directory (using forward slashes)
    std::string binpathkey = "%%ldcbinarypath%%";

    std::string binpath = exe_path::getBinDir();
    std::replace(binpath.begin(), binpath.end(), '\\', '/');

    int len = config_setting_length(sw);
    for (int i = 0; i < len; i++) {
      std::string v(config_setting_get_string(config_setting_get_elem(sw, i)));

      size_t p;
      while (std::string::npos != (p = v.find(binpathkey))) {
        v.replace(p, binpathkey.size(), binpath);
      }

      switches.push_back(strdup(v.c_str()));
    }
  }

  return true;
}
