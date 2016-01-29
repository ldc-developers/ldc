#!/usr/bin/env python
# wrapper to run lit from commandline
if __name__=='__main__':
    try:
        import lit
    except ImportError:
        import sys
        sys.exit('Package lit cannot be imported.\n' \
                 'Lit can be installed using: \'python -m pip install lit\'\n' \
                 '(Python versions older than 2.7.9 or 3.4 do not have pip installed, see:\n' \
                 'https://pip.pypa.io/en/latest/installing/)')

    lit.main()
