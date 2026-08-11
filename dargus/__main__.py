"""Entry point for ``python -m dargus`` — same CLI as the ``dargus`` script."""

import sys

from dargus.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
