import argparse
from . import __version__

def main():
    parser = argparse.ArgumentParser(description="Mini Distributed AI Compiler (scaffold)")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    parser.print_help()

if __name__ == "__main__":
    main()
