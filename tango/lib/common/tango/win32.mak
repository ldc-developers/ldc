# Makefile to build the common D runtime library for Win32
# Designed to work with DigitalMars make
# Targets:
#	make
#		Same as make all
#	make lib
#		Build the common library
#   make doc
#       Generate documentation
#	make clean
#		Delete unneeded files created by build process

LIB_TARGET=tango-cc-tango.lib
LIB_MASK=tango-cc-tango*.lib

CP=xcopy /y
RM=del /f
MD=mkdir

ADD_CFLAGS=
ADD_DFLAGS=

CFLAGS=-mn -6 -r $(ADD_CFLAGS)
#CFLAGS=-g -mn -6 -r $(ADD_CFLAGS)

DFLAGS=-release -O -inline -w -nofloat $(ADD_DFLAGS)
#DFLAGS=-g -w -nofloat $(ADD_DFLAGS)

TFLAGS=-O -inline -w  -nofloat $(ADD_DFLAGS)
#TFLAGS=-g -w -nofloat $(ADD_DFLAGS)

DOCFLAGS=-version=DDoc

CC=dmc
LC=lib
DC=dmd

INC_DEST=..\..\..\tango
LIB_DEST=..
DOC_DEST=..\..\..\doc\tango

.DEFAULT: .asm .c .cpp .d .html .obj

.asm.obj:
	$(CC) -c $<

.c.obj:
	$(CC) -c $(CFLAGS) $< -o$@

.cpp.obj:
	$(CC) -c $(CFLAGS) $< -o$@

.d.obj:
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
    core\BitManip.obj \
    core\Exception.obj \
    core\Memory.obj \
    core\Runtime.obj \
    core\Thread.obj

OBJ_STDC= \
    stdc\wrap.obj

ALL_OBJS= \
    $(OBJ_CORE) \
    $(OBJ_STDC)

######################################################

DOC_CORE= \
    core\BitManip.html \
    core\Exception.html \
    core\Memory.html \
    core\Runtime.html \
    core\Thread.html

ALL_DOCS=

######################################################

tango.lib : $(LIB_TARGET)

$(LIB_TARGET) : $(ALL_OBJS)
	$(RM) $@
	$(LC) -c -n $@ $(ALL_OBJS)

tango.doc : $(ALL_DOCS)
	@echo Documentation generated.

######################################################

### config

# config.obj : config.d
#	$(DC) -c $(DFLAGS) config.d -of$@

######################################################

clean :
	$(RM) /s .\*.di
	$(RM) $(ALL_OBJS)
	$(RM) $(ALL_DOCS)
	$(RM) $(LIB_MASK)

install :
	$(MD) $(INC_DEST)
	$(CP) /s *.di $(INC_DEST)\.
	$(MD) $(DOC_DEST)
	$(CP) /s *.html $(DOC_DEST)\.
	$(MD) $(LIB_DEST)
	$(CP) $(LIB_MASK) $(LIB_DEST)\.
