#!/usr/bin/env bash

set -euo pipefail

if [[ $# -le 3 ]]; then
    echo "Usage: ${0} <input_conf> <output_conf> <placeholder> <ldc2_bin> [<extra_ldc2_args>...]"
    exit 1
fi

input=$1
shift
output=$1
shift
placeholder=$1
shift
ldc2=("${@}")

set +e
triple="$(echo 'module object;' | "${ldc2[@]}" -v -o- - | grep -m 1 '^config')"
set -e
triple="${triple##* (}" # `i686-unknown-linux-gnu)`
triple="${triple%)}"    # `i686-unknown-linux-gnu`
if [ "${triple//[^-]/}" = "---" ]; then
    # ignore vendor => `i686-.*-linux-gnu`
    triple="$(echo "$triple" | sed -e 's/-[^-]*/-.*/')"
fi

if [[ ! ${triple} ]]; then
    echo "Error: failed to parse triple from \"${ldc2[@]} -v\" output"
    exit 1
fi

sed -e "s/${placeholder}/${triple}/" "${input}" > "${output}"
