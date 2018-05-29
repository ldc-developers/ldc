//===-- driver/configfile.h - LDC config file handling ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles reading and parsing of an LDC config file (ldc.conf/ldc2.conf).
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_CONFIGFILE_H
#define LDC_DRIVER_CONFIGFILE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

#include "array.h"

class ConfigFile {
public:
  static ConfigFile instance;

  bool read(const char *explicitConfFile, const char *triple);

  llvm::StringRef path() {
    return pathcstr ? llvm::StringRef(pathcstr) : llvm::StringRef();
  }

  void extendCommandLine(llvm::SmallVectorImpl<const char *> &args);

  llvm::StringRef rpath() {
    return rpathcstr ? llvm::StringRef(rpathcstr) : llvm::StringRef();
  }

private:
  bool locate(std::string &pathstr);
  static bool sectionMatches(const char *section, const char *triple);

  // implemented in D
  bool readConfig(const char *cfPath, const char *triple, const char *binDir);

  const char *pathcstr = nullptr;
  Array<const char *> switches;
  Array<const char *> postSwitches;
  const char *rpathcstr = nullptr;
};

#endif // LDC_DRIVER_CONFIGFILE_H
