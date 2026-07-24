#!/usr/bin/env python3
"""Write a response file with LF-only line endings (dub-style on Windows)."""
import sys


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: write_lf_rsp.py <path> <line>...\n")
        return 1
    path = sys.argv[1]
    content = "\n".join(sys.argv[2:]) + "\n"
    with open(path, "wb") as f:
        f.write(content.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
