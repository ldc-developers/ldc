#pragma once

#include "dmd/common/outbuffer.h"
#include "dmd/globals.h"
#include "dmd/root/filename.h"

FileName runCPreprocessor(FileName csrcfile, const Loc &loc, bool &ifile,
                          OutBuffer &defines);
