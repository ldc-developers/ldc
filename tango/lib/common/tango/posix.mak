# Makefile to build the common D runtime library for Linux
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

CP=cp -f
RM=rm -f
MD=mkdir -p

ADD_CFLAGS=
ADD_DFLAGS=

CFLAGS=-O -m32 $(ADD_CFLAGS)
#CFLAGS=-g -m32 $(ADD_CFLAGS)

DFLAGS=-release -O -inline -w -nofloat -version=Posix $(ADD_DFLAGS)
#DFLAGS=-g -w -nofloat -version=Posix $(ADD_DFLAGS)

TFLAGS=-O -inline -w -nofloat -version=Posix $(ADD_DFLAGS)
#TFLAGS=-g -w -nofloat -version=Posix $(ADD_DFLAGS)

DOCFLAGS=-version=DDoc -version=Posix

CC=gcc
LC=$(AR) -qsv
DC=dmd

INC_DEST=../../../tango
LIB_DEST=..
DOC_DEST=../../../doc/tango

.SUFFIXES: .s .S .c .cpp .d .html .o

.s.o:
	$(CC) -c $(CFLAGS) $< -o$@

.S.o:
	$(CC) -c $(CFLAGS) $< -o$@

.c.o:
	$(CC) -c $(CFLAGS) $< -o$@

.cpp.o:
	g++ -c $(CFLAGS) $< -o$@

.d.o:
	$(DC) -c $(DFLAGS) -Hf$*.di $< -of$@
#	$(DC) -c $(DFLAGS) $< -of$@

.d.html:
	$(DC) -c -o- $(DOCFLAGS) -Df$*.html $<
#	$(DC) -c -o- $(DOCFLAGS) -Df$*.html tango.ddoc $<

targets : lib doc
all     : lib doc
tango   : lib
lib     : tango.lib
doc     : tango.doc

######################################################

OBJ_CORE= \
    core/BitManip.o \
    core/Exception.o \
    core/Memory.o \
    core/Runtime.o \
    core/Thread.o \
    core/ThreadASM.o

OBJ_STDC= \
    stdc/wrap.o

OBJ_STDC_POSIX= \
    stdc/posix/pthread_darwin.o

ALL_OBJS= \
    $(OBJ_CORE) \
    $(OBJ_STDC) \
    $(OBJ_STDC_POSIX)

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

tango.doc : $(ALL_DOCS)
	echo Documentation generated.

######################################################

### stdc/posix

stdc/posix/pthread_darwin.o : stdc/posix/pthread_darwin.d
	$(DC) -c $(DFLAGS) stdc/posix/pthread_darwin.d -of$@

######################################################

clean :
	find . -name "*.di" | xargs $(RM)
	$(RM) $(ALL_OBJS)
	$(RM) $(ALL_DOCS)
	find . -name "$(LIB_MASK)" | xargs $(RM)

install :
	$(MD) $(INC_DEST)
	find . -name "*.di" -exec cp -f {} $(INC_DEST)/{} \;
	$(MD) $(DOC_DEST)
	find . -name "*.html" -exec cp -f {} $(DOC_DEST)/{} \;
	$(MD) $(LIB_DEST)
	find . -name "$(LIB_MASK)" -exec cp -f {} $(LIB_DEST)/{} \;
