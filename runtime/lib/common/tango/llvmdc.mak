# Makefile to build the common D runtime library for LLVM
# Designed to work with GNU make
# Targets:
#	make
#		Same as make all
#	make lib
#		Build the common library
#   make doc
#       Generate documentation
#	make clean
#		Delete unneeded files created by build process

LIB_TARGET=libtango-cc-tango.a
LIB_MASK=libtango-cc-tango*.a
LIB_TARGET_C=libtango-cc-c-tango.a
LIB_MASK_C=libtango-cc-c-tango*.a

CP=cp -f
RM=rm -f
MD=mkdir -p

ADD_CFLAGS=
ADD_DFLAGS=

#CFLAGS=-O3 $(ADD_CFLAGS)
CFLAGS=$(ADD_CFLAGS)

#DFLAGS=-release -O3 -inline -w $(ADD_DFLAGS)
DFLAGS=-w -noasm $(ADD_DFLAGS)

#TFLAGS=-O3 -inline -w $(ADD_DFLAGS)
TFLAGS=-w -noasm $(ADD_DFLAGS)

DOCFLAGS=-version=DDoc

CC=gcc
LC=llvm-ar rsv
CLC=ar rsv
DC=llvmdc
LLC=llvm-as

INC_DEST=../../../tango
LIB_DEST=..
DOC_DEST=../../../doc/tango

.SUFFIXES: .s .S .c .cpp .d .ll .html .o .bc

.s.o:
	$(CC) -c $(CFLAGS) $< -o$@

.S.o:
	$(CC) -c $(CFLAGS) $< -o$@

.c.o:
	$(CC) -c $(CFLAGS) $< -o$@

.cpp.o:
	g++ -c $(CFLAGS) $< -o$@

.d.bc:
	$(DC) -c $(DFLAGS) -Hf$*.di $< -of$@
#	$(DC) -c $(DFLAGS) $< -of$@

.ll.bc:
	$(LLC) -f -o=$@ $<

.d.html:
	$(DC) -c -o- $(DOCFLAGS) -Df$*.html $<
#	$(DC) -c -o- $(DOCFLAGS) -Df$*.html tango.ddoc $<

targets : lib doc
all     : lib doc
tango   : lib
lib     : tango.lib tango.clib
doc     : tango.doc

######################################################

OBJ_CORE= \
    core/BitManip.bc \
    core/Exception.bc \
    core/Memory.bc \
    core/Runtime.bc \
    core/Thread.bc
#    core/ThreadASM.o

OBJ_STDC= \
    stdc/wrap.o
#    stdc/wrap.bc

OBJ_STDC_POSIX= \
    stdc/posix/pthread_darwin.o

ALL_OBJS= \
    $(OBJ_CORE)
#    $(OBJ_STDC)
#    $(OBJ_STDC_POSIX)

######################################################

DOC_CORE= \
    core/BitManip.html \
    core/Exception.html \
    core/Memory.html \
    core/Runtime.html \
    core/Thread.html


ALL_DOCS=

######################################################

tango.lib : $(LIB_TARGET)

$(LIB_TARGET) : $(ALL_OBJS)
	$(RM) $@
	$(LC) $@ $(ALL_OBJS)


tango.clib : $(LIB_TARGET_C)

$(LIB_TARGET_C) : $(OBJ_STDC)
	$(RM) $@
	$(CLC) $@ $(OBJ_STDC)


tango.doc : $(ALL_DOCS)
	echo Documentation generated.

######################################################

### stdc/posix

#stdc/posix/pthread_darwin.o : stdc/posix/pthread_darwin.d
#	$(DC) -c $(DFLAGS) stdc/posix/pthread_darwin.d -of$@

######################################################

clean :
	find . -name "*.di" | xargs $(RM)
	$(RM) $(ALL_OBJS)
	$(RM) $(OBJ_STDC)
	$(RM) $(ALL_DOCS)
	find . -name "$(LIB_MASK)" | xargs $(RM)
	find . -name "$(LIB_MASK_C)" | xargs $(RM)

install :
	$(MD) $(INC_DEST)
	find . -name "*.di" -exec cp -f {} $(INC_DEST)/{} \;
	$(MD) $(DOC_DEST)
	find . -name "*.html" -exec cp -f {} $(DOC_DEST)/{} \;
	$(MD) $(LIB_DEST)
	find . -name "$(LIB_MASK)" -exec cp -f {} $(LIB_DEST)/{} \;
	find . -name "$(LIB_MASK_C)" -exec cp -f {} $(LIB_DEST)/{} \;
