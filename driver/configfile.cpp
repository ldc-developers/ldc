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
#include "llvm/Support/CommandLine.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

namespace sys = llvm::sys;

// dummy only; needs to be parsed manually earlier as the switches contained in
// the config file are injected into the command line options fed to the parser
llvm::cl::opt<std::string>
    clConf("conf", llvm::cl::desc("Use configuration file <filename>"),
           llvm::cl::value_desc("filename"));


const char *getLdcInstallPrefixCStr()
{
#define STR(x) #x
#define XSTR(x) STR(x)

  return XSTR(LDC_INSTALL_PREFIX);

#undef XSTR
#undef STR
}
