# Makefile to build the garbage collector D library for LDC
# Designed to work with GNU make
# Targets:
#	make
#		Same as make all
#	make lib
#		Build the garbage collector library
#   make doc
#       Generate documentation
#	make clean
#		Delete unneeded files created by build process

LIB_TARGET_BC=libtango-gc-basic-bc.a
LIB_TARGET_NATIVE=libtango-gc-basic.a
LIB_TARGET_SHARED=libtango-gc-basic-shared.so
LIB_MASK=libtango-gc-basic*.*

CP=cp -f
RM=rm -f
MD=mkdir -p

ADD_CFLAGS=
ADD_DFLAGS=

#CFLAGS=-O3 $(ADD_CFLAGS)
CFLAGS=$(ADD_CFLAGS)

#DFLAGS=-release -O3 -inline -w -nofloat $(ADD_DFLAGS)
DFLAGS=-w -disable-invariants $(ADD_DFLAGS)

#TFLAGS=-O3 -inline -w -nofloat $(ADD_DFLAGS)
TFLAGS=-w -disable-invariants $(ADD_DFLAGS)

DOCFLAGS=-version=DDoc

CC=gcc
LC=llvm-ar rsv
LCC=llc
LLINK=llvm-link
CLC=ar rsv
LD=llvm-ld
DC=ldc

LIB_DEST=..

.SUFFIXES: .s .S .c .cpp .d .html .o .bc

.s.o:
	$(CC) -c $(CFLAGS) $< -o$@

.S.o:
	$(CC) -c $(CFLAGS) $< -o$@

.c.o:
	$(CC) -c $(CFLAGS) $< -o$@

.cpp.o:
	g++ -c $(CFLAGS) $< -o$@

.d.bc:
	$(DC) -c $(DFLAGS) $< -of$@ -output-bc

.d.o:
	$(DC) -c $(DFLAGS) $< -of$@

.d.html:
	$(DC) -c -o- $(DOCFLAGS) -Df$*.html $<
#	$(DC) -c -o- $(DOCFLAGS) -Df$*.html dmd.ddoc $<

targets : lib sharedlib doc
all     : lib sharedlib doc
lib     : basic.lib basic.nlib
sharedlib : basic.sharedlib
doc     : basic.doc

######################################################

ALL_OBJS_BC= \
    gc.bc \
    gcalloc.bc \
    gcbits.bc \
    gcstats.bc \
    gcx.bc

ALL_OBJS_O= \
    gc.o \
    gcalloc.o \
    gcbits.o \
    gcstats.o \
    gcx.o

######################################################

ALL_DOCS=

######################################################

basic.lib : $(LIB_TARGET_BC)
basic.nlib : $(LIB_TARGET_NATIVE)
basic.sharedlib : $(LIB_TARGET_SHARED)

$(LIB_TARGET_BC) : $(ALL_OBJS_BC)
	$(RM) $@
	$(LC) $@ $(ALL_OBJS_BC)


$(LIB_TARGET_NATIVE) : $(ALL_OBJS_O)
	$(RM) $@
	$(CLC) $@ $(ALL_OBJS_O)


$(LIB_TARGET_SHARED) : $(ALL_OBJS_O)
	$(RM) $@
	$(CC) -shared -o $@ $(ALL_OBJS_O)

basic.doc : $(ALL_DOCS)
	echo No documentation available.

######################################################

clean :
	find . -name "*.di" | xargs $(RM)
	$(RM) $(ALL_OBJS)
	$(RM) $(ALL_DOCS)
	$(RM) $(LIB_MASK)

install :
	$(MD) $(LIB_DEST)
	$(CP) $(LIB_MASK) $(LIB_DEST)/.
