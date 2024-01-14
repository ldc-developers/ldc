# set explicitly in the make cmdline in druntime/Makefile (`test/%/.run` rule):
ifneq (,$(findstring ldmd2,$(DMD)))
    # LDC: we have no top makefile, include osmodel.mak for OS and set up bash shell on Windows
    MODEL:=default
    include ../../../../dmd/osmodel.mak
else
    OS:=
    MODEL:=
endif
BUILD:=
DMD:=
DRUNTIME:=
DRUNTIMESO:=
LINKDL:=
QUIET:=
TIMELIMIT:=
PIC:=

ifeq (,$(findstring ldmd2,$(DMD)))
    # Windows: set up bash shell
    ifeq (windows,$(OS))
        include ../../../compiler/src/osmodel.mak
    endif
endif

LDL:=$(subst -L,,$(LINKDL)) # -ldl
SRC:=src
GENERATED:=./generated
ROOT:=$(GENERATED)/$(OS)/$(BUILD)/$(MODEL)
DRUNTIME_IMPLIB:=$(subst .dll,.lib,$(DRUNTIMESO))

MODEL_FLAG:=$(if $(findstring $(MODEL),default),,-m$(MODEL))
CFLAGS_BASE:=$(if $(findstring $(OS),windows),/Wall,$(MODEL_FLAG) $(PIC) -Wall)
ifeq (,$(findstring ldmd2,$(DMD)))
    ifeq (osx64,$(OS)$(MODEL))
        CFLAGS_BASE+=--target=x86_64-darwin-apple  # ARM cpu is not supported by dmd
    endif
endif
# LDC: use `-defaultlib=druntime-ldc [-link-defaultlib-shared]` instead of `-defaultlib= -L$(DRUNTIME[_IMPLIB])`
DFLAGS:=$(MODEL_FLAG) $(PIC) -w -I../../src -I../../import -I$(SRC) -defaultlib=$(if $(findstring ldmd2,$(DMD)),druntime-ldc,) -preview=dip1000 $(if $(findstring $(OS),windows),,-L-lpthread -L-lm $(LINKDL))
# LINK_SHARED may be set by importing makefile
ifeq (,$(findstring ldmd2,$(DMD)))
    DFLAGS+=$(if $(LINK_SHARED),-L$(DRUNTIME_IMPLIB) $(if $(findstring $(OS),windows),-dllimport=defaultLibsOnly),-L$(DRUNTIME))
else
    # LDC: -link-defaultlib-shared takes care of rpath, linking ldc_rt.dso.o etc.
    DFLAGS+=$(if $(LINK_SHARED),-link-defaultlib-shared,)
endif
ifeq ($(BUILD),debug)
    DFLAGS+=-g -debug $(if $(findstring ldmd2,$(DMD)),-link-defaultlib-debug,)
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
