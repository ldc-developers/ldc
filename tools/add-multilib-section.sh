#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <path/to/ldc2> <32|64> <path/to/multilib-dir> <rpath> <path/to/input.conf> <path/to/output.conf>"
  exit 1
fi

ldc=$1
bitness=$2
libdir=$3
rpath=$4
input_conf=$5
output_conf=$6

# parse multilib triple from `-v` output
set +e
# extract `config    /path/to/ldc2.conf (i686-unknown-linux-gnu)`
triple="$(echo "module object;" | $ldc -m$bitness -v -o- -conf=$input_conf - | grep -m 1 '^config')"
triple="${triple##* (}" # `i686-unknown-linux-gnu)`
triple="${triple%)}"    # `i686-unknown-linux-gnu`
if [ "${triple//[^-]/}" = "---" ]; then
  # ignore vendor => `i686-.*-linux-gnu`
  triple="$(echo "$triple" | sed -e 's/-[^-]*/-.*/')"
fi
set -e

cp $input_conf $output_conf

if [ -z "$triple" ]; then
  echo "Error: failed to parse triple from \"$ldc -m$bitness -v\" output"
  exit 1
fi

# append config file section
section="
\"$triple\":
{
    lib-dirs = [
        \"$libdir\",
    ];
    rpath = \"$rpath\";
};"
echo "$section" >> $output_conf
