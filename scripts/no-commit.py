#!/usr/bin/env python
import argparse
import sys

from pre_commit_hooks import files_contain_tag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='NOCOMMIT')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    if files_contain_tag(args.files, tag=args.tag):
        sys.exit(1)
