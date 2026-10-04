#!/usr/bin/env bash
# Build the nPrint v1.2.1 CLI that FlowPaint's Traffic Encoder / Protocol Decoder
# depend on. nPrint is a third-party tool (Apache-2.0) by Holland et al. (nprintML,
# IMC'21). This script fetches and builds it into third_party/nprint-1.2.1/.
#
# After building, either add it to your PATH or point NPRINT_BIN at the binary:
#   export NPRINT_BIN="$(pwd)/third_party/nprint-1.2.1/nprint"
#
# Requires: git, autoconf/automake/libtool, a C++ compiler, and libpcap headers
# (Debian/Ubuntu: sudo apt-get install build-essential autoconf automake libtool libpcap-dev).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/third_party"
mkdir -p "${DEST}"
cd "${DEST}"

if [ ! -d nprint-1.2.1 ]; then
  git clone --branch v1.2.1 --depth 1 https://github.com/nprint/nprint.git nprint-src || \
  git clone --depth 1 https://github.com/nprint/nprint.git nprint-src
  mv nprint-src nprint-1.2.1
fi

cd nprint-1.2.1
./autogen.sh 2>/dev/null || autoreconf -fi
./configure
make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

echo
echo "Built nprint at: ${DEST}/nprint-1.2.1/nprint"
echo "Set: export NPRINT_BIN=${DEST}/nprint-1.2.1/nprint"
