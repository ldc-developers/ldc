
# ROOT_DIR = directory where Makefile sits
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(dir $(MAKEFILE_PATH))

LLVM_CONFIG ?= llvm-config

CXXFLAGS ?= -O3
CXXFLAGS += $(shell $(LLVM_CONFIG) --cxxflags) -fno-rtti -fpic
# Remove all warning flags (they may or may not be supported by the compiler)
CXXFLAGS := $(filter-out -W%,$(CXXFLAGS))
CXXFLAGS := $(filter-out -fcolor-diagnostics,$(CXXFLAGS))

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
