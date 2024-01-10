# set explicitly in the make cmdline in druntime/Makefile (`test/%/.run` rule):
# LDC: we have no top makefile, include osmodel.mak for OS and set up bash shell
include ../../../../dmd/osmodel.mak
#OS:=
MODEL:=
BUILD:=
DMD:=
DRUNTIME:=
DRUNTIMESO:=
LINKDL:=
QUIET:=
TIMELIMIT:=
PIC:=

LDL:=$(subst -L,,$(LINKDL)) # -ldl
SRC:=src
GENERATED:=./generated
ROOT:=$(GENERATED)/$(OS)/$(BUILD)/$(MODEL)
DRUNTIME_IMPLIB:=$(subst .dll,.lib,$(DRUNTIMESO))

MODEL_FLAG:=$(if $(findstring $(MODEL),default),,-m$(MODEL))
CFLAGS_BASE:=$(if $(findstring $(OS),windows),/Wall,$(MODEL_FLAG) $(PIC) -Wall)
#ifeq (osx64,$(OS)$(MODEL))
#    CFLAGS_BASE+=--target=x86_64-darwin-apple  # ARM cpu is not supported by dmd
#endif
# LDC: use `-defaultlib=druntime-ldc [-link-defaultlib-shared]` instead of `-defaultlib= -L$(DRUNTIME[_IMPLIB])`
DFLAGS:=$(MODEL_FLAG) $(PIC) -w -I../../src -I../../import -I$(SRC) -defaultlib=druntime-ldc -preview=dip1000 $(if $(findstring $(OS),windows),,-L-lpthread -L-lm $(LINKDL))
# LINK_SHARED may be set by importing makefile
# LDC: -link-defaultlib-shared takes care of rpath, linking ldc_rt.dso.o etc.
DFLAGS+=$(if $(LINK_SHARED),-link-defaultlib-shared,)
ifeq ($(BUILD),debug)
    # LDC: link against debug druntime
    DFLAGS+=-g -debug -link-defaultlib-debug
    CFLAGS:=$(CFLAGS_BASE) $(if $(findstring $(OS),windows),/Zi,-g)
else
    DFLAGS+=-O -release
    CFLAGS:=$(CFLAGS_BASE) $(if $(findstring $(OS),windows),/O2,-O3)
endif
CXXFLAGS_BASE:=$(CFLAGS_BASE)
CXXFLAGS:=$(CFLAGS)

ifeq (windows,$(OS))
    DOTEXE:=.exe
    DOTDLL:=.dll
    DOTLIB:=.lib
    DOTOBJ:=.obj
else
    DOTEXE:=
    DOTDLL:=$(if $(findstring $(OS),osx),.dylib,.so)
    DOTLIB:=.a
    DOTOBJ:=.o
endif
