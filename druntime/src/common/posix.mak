# Makefile to build the D runtime library core components for Posix
# Designed to work with GNU make
# Targets:
#	make
#		Same as make all
#	make debug
#		Build the debug version of the library
#   make release
#       Build the release version of the library
#   make doc
#       Generate documentation
#	make clean
#		Delete all files created by build process

# Essentials

LIBDIR=../../lib
DOCDIR=../../doc
IMPDIR=../../import
LIBBASENAME=libdruntime-core.a
MODULES=bitop exception memory runtime thread vararg \
	$(addprefix sync/,barrier condition config exception mutex rwmutex semaphore)
BUILDS=debug release unittest

# Symbols

DMD=dmd
CLC=ar rsv
DOCFLAGS=-version=DDoc
DFLAGS_release=-d -release -O -inline -w
DFLAGS_debug=-d -g -w
DFLAGS_unittest=$(DFLAGS_release) -unittest
CFLAGS_release= -O
CFLAGS_debug= -g
CFLAGS_unittest=$(CFLAGS_release)

# Derived symbols

C_SRCS=core/stdc/errno.c #core/threadasm.S
C_OBJS=errno.o threadasm.o
AS_OBJS=$(addsuffix .o,$(basename $(AS_SRCS)))
D_SRCS=$(addsuffix .d,$(addprefix core/,$(MODULES))) \
	$(addsuffix .d,$(addprefix $(IMPDIR)/core/stdc/,math stdarg stdio wchar_)) \
	$(addsuffix .d,$(addprefix $(IMPDIR)/core/sys/posix/,netinet/in_ sys/select sys/socket sys/stat sys/wait))
ALL_OBJS_O=$(addsuffix .o,$(addprefix core/,$(MODULES))) \
	$(addsuffix .o,$(addprefix $(IMPDIR)/core/stdc/,math stdarg stdio wchar_)) \
	$(addsuffix .o,$(addprefix $(IMPDIR)/core/sys/posix/,netinet/in_ sys/select sys/socket sys/stat sys/wait)) \
	$(AS_OBJS) $(C_OBJS)
DOCS=$(addsuffix .html,$(addprefix $(DOCDIR)/core/,$(MODULES)))
IMPORTS=$(addsuffix .di,$(addprefix $(IMPDIR)/core/,$(MODULES)))
ALLLIBS=$(addsuffix /$(LIBBASENAME),$(addprefix $(LIBDIR)/,$(BUILDS)))

# Patterns

$(LIBDIR)/%/$(LIBBASENAME) : $(D_SRCS) $(C_SRCS)
	$(CC) -c $(CFLAGS_$*) $(C_SRCS)
ifeq ($(DMD),ldc2 -vv)
	$(DMD) $(DFLAGS_$*) -of$@ $(D_SRCS)
	$(CLC) $@ $(ALL_OBJS_O)
else
	$(DMD) $(DFLAGS_$*) -lib -of$@ $(D_SRCS) $(C_OBJS)
endif
	rm $(C_OBJS)

$(DOCDIR)/%.html : %.d
	$(DMD) -c -d -o- -Df$@ $<

$(IMPDIR)/%.di : %.d
	$(DMD) -c -d -o- -Hf$@ $<

# Rulez

all : $(BUILDS) doc

debug : $(LIBDIR)/debug/$(LIBBASENAME) $(IMPORTS)
release : $(LIBDIR)/release/$(LIBBASENAME) $(IMPORTS)
unittest : $(LIBDIR)/unittest/$(LIBBASENAME) $(IMPORTS)
doc : $(DOCS)

clean :
	rm -f $(IMPORTS) $(DOCS) $(ALLLIBS)
