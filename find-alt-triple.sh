#!/bin/bash

HOST_TARGET=$1

[ "$HOST_TARGET" = "" ] && exit 1

REST=`echo $HOST_TARGET   | sed -e 's/[a-zA-Z0-9_]*\(\-.*\)/\1/'`
X86=`echo $HOST_TARGET    | sed -e 's/\(i[3-9]86\)\-.*/\1/'`
X86_64=`echo $HOST_TARGET | sed -e 's/\(x86_64\)\-.*/\1/'`

ALT=
if [ "$X86_64" != "$HOST_TARGET" ]; then
    ALT="i686$REST"

elif [ "$X86" != "$HOST_TARGET" ]; then
    ALT="x86_64$REST"
fi

echo $ALT
