#!/usr/bin/env bash
set -euxo pipefail

# This script creates a fresh temporary clone of the DMD repo and rewrites
# it (via `git filter-repo`) according to LDC requirements (removing
# unnecessary files and transforming the directory structure).
# Then it pushes new tags and updated master/stable/v2.* branches (all prefixed
# with `dmd-rewrite-`) to the LDC repo, i.e., the rewritten updates to
# the DMD repo since the last successful script invocation.
#
# Afterwards, merging a new frontend+druntime is a matter of merging branch
# `dmd-rewrite-stable` or some tag (`dmd-rewrite-v2.101.0`).
#
# NOTE: As it turns out, the rewrite unfortunately isn't stable across git versions.
#       This process was started on Ubuntu 22, using git v2.34 and git-filter-repo v2.34 (22826b5a68b6).
#       On Ubuntu 24 (git v2.43), this fails (to fast-forward push) with both default git-filter-repo v2.38 (ed61b4050b71) and v2.34.
#       So you might need to use a Ubuntu 22 container for now to run this script successfully.

initialize="${INITIALIZE:-0}" # set INITIALIZE=1 for the very first rewrite
refs_prefix="dmd-rewrite-"
branches=(master stable v2.113 v2.114)

temp_dir="$(mktemp -d)"
cd "$temp_dir"

# generate message-filters.txt file for replacing GitHub refs in commit messages
cat > message-filters.txt <<'EOF'
regex:(^|\s|\()#(\d{2,})==>\1dlang/dmd!\2
(cherry picked from commit 88d1e8fc37428b873f59d87f8dff1f40fbd3e7a3)==>(cherry picked from commit 8b9b481a322bdcbfdad38ba4ad74182742aef118)
(cherry picked from commit bd8816e5d9126b88494e0d10028383812ec91a86)==>(cherry picked from commit 86d950c26d19102a7df14a14b577c90cce383563)
(cherry picked from commit 1f8b722912480531f9c68fef99b20ee018d55814)==>(cherry picked from commit 53bf64bd002b4596a91514dda139d72d57ac940b)
(cherry picked from commit 7e0b115b870b08d0a4658b487471ce3bebb79281)==>(cherry picked from commit edb4c9ffc0daa00f72e21ab0847869d8dee779fa)
(cherry picked from commit 579fe56f86df2202ea6fd5347e65aeb8f6421a15)==>(cherry picked from commit ab34ae7b403fa8d3d53eb07c28a152e13083a466)
(cherry picked from commit 403a9ff66c3e517b4d8d84c555f7f32f67f99c61)==>(cherry picked from commit 297d8f4afcfa9eb9cdba35d793f46272c8884a95)
(cherry picked from commit ff7c22d91fbd88fd021e1397fbc9813c5f1a59fc)==>(cherry picked from commit 01462da180ca7b349d123f4ba0b1c7d3de1a0085)
(cherry picked from commit 0c0d44aaf87c3e53b9dfd1b26baeef147b225ef1)==>(cherry picked from commit 832673894b2f73ac83c318bcd5e05ef38800736d)
(cherry picked from commit 5100a7df7e26aa1411ed724d61276730967aa1ca)==>(cherry picked from commit 86a91cccd4338a0e972f6f73587dfd6f2cdf0849)
(cherry picked from commit 95711eea4072dd9e0b74a1711ca70f2d9263561e)==>(cherry picked from commit 2e88540e7a929b8161c7f8497130201a376f1f0f)
(cherry picked from commit c15569281ba36901f4a2d556d736f6d000109887)==>(cherry picked from commit f2717909feca9dd31f823dc1508bfaa70f775ff1)
(cherry picked from commit 10323166d86b9519c41a13c1a3bdf4a469562eb1)==>(cherry picked from commit 2863ebeaf853b0f9a39fb541aecd53c89cca5d93)
(added by 53e3cdde4b,==>(added by 504c42f405,
(cherry picked from commit b892690c39b9ee7192ec9c445dd4c8f217803f80)==>(cherry picked from commit adaa4c7734109de675a844a8748e02bc2543c7c6)
This reverts commit fa1f860e4be6a0d796a47329be110be14fc1d667==>This reverts commit 601bef533244166f1f114d424e6d3b3f8119d697
This reverts commit 4b76bfcb2eb24bb8786bd293201a5711fbf747fe==>This reverts commit 601bef533244166f1f114d424e6d3b3f8119d697
EOF

# clone DMD monorepo
git clone https://github.com/dlang/dmd.git dmd.tmp
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
  --path compiler/src/dmd --path compiler/include/dmd --path compiler/test --path compiler/src/osmodel.mak \
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
  --path 'compiler/src/dmd/lib/' \
  --path-regex '(compiler/)?src/dmd/link\.d' \
  --path-regex '(compiler/)?src/dmd/(e|s)2ir\.d' \
  --path-regex '(compiler/)?src/dmd/to(csym|ctype|cvdebug|dt|ir|obj)\.d' \
  --path-regex '(compiler/)?src/dmd/(objc_)?glue\.d' \
  --path 'compiler/src/dmd/glue/' \
  --path-regex '(compiler/)?src/dmd/iasm(|dmd)\.d' \
  --path-regex 'compiler/src/dmd/iasm/(package|dmd(aarch64|x86))\.d' \
  --path-glob 'src/*.mak' # remaining Makefiles after moving dmd into compiler/
git filter-repo \
  `# move dirs/files` \
  --path-rename druntime/:runtime/druntime/ \
  --path-rename compiler/src/dmd/:dmd/ \
  --path-rename compiler/include/dmd/:dmd/ \
  --path-rename compiler/test/:tests/dmd/ \
  --path-rename compiler/src/osmodel.mak:dmd/osmodel.mak \
  `# prefix tags` \
  --tag-rename ":$refs_prefix" \
  `# replace GitHub refs in commit messages` \
  --replace-message ../message-filters.txt

# create prefixed branches
for b in ${branches[@]}; do
  git branch --no-track "${refs_prefix}$b" $b
done

# add LDC repo as `ldc` remote
if [[ -v GITHUB_TOKEN ]]; then
  git remote add ldc https://git:${GITHUB_TOKEN}@github.com/ldc-developers/ldc.git
else
  git remote add ldc git@github.com:ldc-developers/ldc.git
fi

if [[ "$initialize" != 1 ]]; then
  # fetch the current rewritten branches
  git fetch ldc ${branches[@]/#/${refs_prefix}}

  # check for newly added DMD source files
  numFiles="$(git diff --name-only --diff-filter=A --relative=dmd "ldc/${refs_prefix}master..${refs_prefix}master" | wc -l)"
  if [[ "$numFiles" != 0 ]]; then
    set +x
    echo "$numFiles newly added DMD source files in branch master:"
    echo "----------"
    git diff --name-only --diff-filter=A --relative=dmd "ldc/${refs_prefix}master..${refs_prefix}master"
    echo "----------"

    proceed=0
    if [[ ! -v GITHUB_EVENT_NAME ]]; then
      read -p "Are *all* of these newly added files of interest to LDC, i.e., should they all be included in the rewrite? [yN] " yn
      if [[ "$yn" == y ]]; then
        proceed=1
        echo "Okay, proceeding with adding these new files for LDC too."
      fi
    fi
    if [[ "$proceed" -ne 1 ]]; then
      echo "Aborting due to these newly added files."
      echo "Please check if they are of interest to LDC:"
      echo "* If they aren't, have them removed via filter-repo in this script, then re-run the script."
      echo "* If they are, re-run this script interactively and confirm the according prompt."
      exit 1
    fi
    set -x
  fi

  # check if all branches can be fast-forwarded
  set +x
  fast_forward=1
  for b in ${branches[@]}; do
    branch="${refs_prefix}${b}"
    if ! git merge-base --is-ancestor ldc/$branch $branch; then
      fast_forward=0
      echo "Error: Branch $branch cannot be fast-forwarded!"
      echo "       The last common ancestor is: $(git merge-base ldc/$branch $branch)"
    fi
  done
  if [[ $fast_forward -ne 1 ]]; then
    echo "Keeping repository in $PWD for manual investigation."
    exit 1
  fi
  set -x
fi

# push prefixed branches and tags (except for PR runs)
if [[ "${GITHUB_EVENT_NAME:-}" != pull_request ]]; then
  git push --tags --atomic ldc ${branches[@]/#/${refs_prefix}}
fi

cd ../..
rm -rf "$temp_dir"
