
# ROOT_DIR = directory where Makefile sits
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(dir $(MAKEFILE_PATH))

LLVM_CONFIG ?= llvm-config

CXXFLAGS ?= -O3
CXXFLAGS += `$(LLVM_CONFIG) --cxxflags` -fno-rtti -fpic

ifeq "$(shell uname)" "Darwin"
  CXXFLAGS += -Wl,-flat_namespace -Wl,-undefined,suppress
endif

PASSLIB = addFuncEntryCallPass

all: $(PASSLIB)

$(PASSLIB): $(ROOT_DIR)$(PASSLIB).cpp
	$(CXX) $(CXXFLAGS) -shared $< -o $@.so

.NOTPARALLEL: clean

clean:
	rm -f $(PASSLIB).so
