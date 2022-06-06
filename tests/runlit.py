#!/usr/bin/env python3
# wrapper to run lit from commandline

from __future__ import print_function

if __name__=='__main__':
    try:
        import lit.main
    except ImportError:
        import sys
        sys.exit('Package lit cannot be imported.\n' \
                 'Lit can be installed using: \'python -m pip install -U lit\'\n' \
                 '(Python versions older than 2.7.9 or 3.4 do not have pip installed, see:\n' \
                 'https://pip.pypa.io/en/latest/installing/)')

    print("Lit version: ", lit.__version__)
    lit.main.main()
