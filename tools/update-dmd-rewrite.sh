#!/usr/bin/env bash
set -euxo pipefail

# This script creates a fresh temporary clone of the DMD repo and rewrites
# it (via `git filter-repo`) according to LDC requirements (removing
# unnecessary files and transforming the directory structure).
# Then it pushes new tags and updated master/stable branches (all prefixed
# with `dmd-rewrite-`) to the LDC repo, i.e., the rewritten updates to
# the DMD repo since the last successful script invocation.
#
# Afterwards, merging a new frontend+druntime is a matter of merging branch
# `dmd-rewrite-stable` or some tag (`dmd-rewrite-v2.101.0`).

initialize="${INITIALIZE:-0}" # set INITIALIZE=1 for the very first rewrite
refs_prefix="dmd-rewrite-"

temp_dir="$(mktemp -d)"
cd "$temp_dir"

# generate message-filters.txt file for replacing GitHub refs in commit messages
echo 'regex:(^|\s|\()#(\d{2,})==>\1dlang/dmd!\2' > message-filters.txt

# clone DMD monorepo
git clone git@github.com:dlang/dmd.git dmd.tmp
cd dmd.tmp

# only for initialization:
# merge LDC druntime patches (tag: first-merged-dmd-druntime-with-ldc-druntime-patches)
if [[ "$initialize" == 1 ]]; then
  git checkout -b ldc-tmp first-merged-dmd-druntime --no-track
  git subtree pull --prefix druntime git@github.com:ldc-developers/druntime.git ldc-pre-monorepo -m "Merge LDC pre-monorepo druntime patches"
  git tag first-merged-dmd-druntime-with-ldc-druntime-patches
fi

# extract a subset of the dmd monorepo (druntime source + tests + Makefiles, dmd source + tests + osmodel.mak)
git filter-repo --force \
  --path druntime/src --path druntime/test --path druntime/Makefile --path-glob 'druntime/*.mak' \
  --path compiler/src/dmd --path compiler/test --path compiler/src/osmodel.mak \
  --path src --path test # required to keep git history before upstream druntime-merge (and associated directory movals)
# remove unused files
git filter-repo --invert-paths \
  --path-regex '(compiler/)?src/dmd/backend' \
  --path-regex '(compiler/)?src/dmd/root/(env|response|strtold)\.d' \
  --path-regex '(compiler/)?src/dmd/astbase\.d' \
  --path-regex '(compiler/)?src/dmd/cpreprocess\.d' \
  --path-regex '(compiler/)?src/dmd/dinifile\.d' \
  --path-regex '(compiler/)?src/dmd/dmdparams\.d' \
  --path-regex '(compiler/)?src/dmd/dmsc\.d' \
  --path-regex '(compiler/)?src/dmd/eh\.d' \
  --path-regex '(compiler/)?src/dmd/frontend\.d' \
  --path-regex '(compiler/)?src/dmd/scan(elf|mach|mscoff|omf)\.d' \
  --path-regex '(compiler/)?src/dmd/lib(|elf|mach|mscoff|omf)\.d' \
  --path-regex '(compiler/)?src/dmd/link\.d' \
  --path-regex '(compiler/)?src/dmd/(e|s)2ir\.d' \
  --path-regex '(compiler/)?src/dmd/to(csym|ctype|cvdebug|dt|ir|obj)\.d' \
  --path-regex '(compiler/)?src/dmd/iasm(|dmd)\.d' \
  --path-regex '(compiler/)?src/dmd/(objc_)?glue\.d' \
  --path-glob 'src/*.mak' # remaining Makefiles after moving dmd into compiler/
git filter-repo \
  `# move dirs/files` \
  --path-rename druntime/:runtime/druntime/ \
  --path-rename compiler/src/dmd/:dmd/ \
  --path-rename compiler/test/:tests/dmd/ \
  --path-rename compiler/src/osmodel.mak:dmd/osmodel.mak \
  `# prefix tags` \
  --tag-rename ":$refs_prefix" \
  `# replace GitHub refs in commit messages` \
  --replace-message ../message-filters.txt

# create prefixed master/stable branches
git branch --no-track "${refs_prefix}master" master
git branch --no-track "${refs_prefix}stable" stable

# add LDC repo as `ldc` remote
git remote add ldc git@github.com:ldc-developers/ldc.git

# check for newly added DMD source files
if [[ "$initialize" != 1 ]]; then
  git fetch ldc "${refs_prefix}master"
  numFiles="$(git diff --name-only --diff-filter=A --relative=dmd "ldc/${refs_prefix}master..${refs_prefix}master" | wc -l)"
  if [[ "$numFiles" != 0 ]]; then
    set +x
    echo "$numFiles newly added DMD source files in branch master:"
    echo "----------"
    git diff --name-only --diff-filter=A --relative=dmd "ldc/${refs_prefix}master..${refs_prefix}master"
    echo "----------"
    if [[ "${FORCE:-0}" == 1 ]]; then
      echo "Adding these files due to env variable FORCE=1."
    else
      echo "Aborting due to these newly added files."
      echo "Please check if they are of interest to LDC:"
      echo "* If they aren't, have them removed via filter-repo in this script, then re-run the script."
      echo "* If they are, re-run this script with env variable FORCE=1."
      exit 1
    fi
    set -x
  fi
fi

# push prefixed master/stable branches and tags
git push --tags ldc "${refs_prefix}master" "${refs_prefix}stable"

cd ../..
rm -rf "$temp_dir"
