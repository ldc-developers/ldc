# set from top makefile
# LDC: we have no top makefile, include osmodel.mak for OS
include ../../../../tests/d2/src/osmodel.mak
#OS:=
MODEL:=
BUILD:=
DMD:=
DRUNTIME:=
DRUNTIMESO:=
LINKDL:=
QUIET:=
TIMELIMIT:=
LDL:=$(subst -L,,$(LINKDL)) # -ldl

SRC:=src
GENERATED:=./generated
ROOT:=$(GENERATED)/$(OS)/$(BUILD)/$(MODEL)

ifneq (default,$(MODEL))
	MODEL_FLAG:=-m$(MODEL)
endif
CFLAGS:=$(MODEL_FLAG) $(PIC) -Wall
DFLAGS:=$(MODEL_FLAG) $(PIC) -w -I../../src -I../../import -I$(SRC) -defaultlib= -debuglib= -dip1000
# LINK_SHARED may be set by importing makefile
# LDC: -link-defaultlib-shared enables default rpath
DFLAGS+=$(if $(LINK_SHARED),-L$(DRUNTIMESO) -link-defaultlib-shared,-L$(DRUNTIME))
ifeq ($(BUILD),debug)
	DFLAGS += -g -debug
	CFLAGS += -g
else
	DFLAGS += -O -release
	CFLAGS += -O3
endif
CXXFLAGS:=$(CFLAGS) -std=c++11
ifeq (osx,$(OS))
	CXXFLAGS+=-stdlib=libc++
endif
