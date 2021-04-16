from typing import Sequence


def file_contains_tag(filename: str, *, tag: str) -> bool:
    with open(filename, 'r') as f:
        lines = f.read().split('\n')

    contains_tag = False
    for i, line in enumerate(lines, start=1):
        if tag in line:
            print(f'{filename}: line {i} contains tag "{tag}"')
            contains_tag = True

    return contains_tag


def files_contain_tag(filenames: Sequence[str], *, tag: str) -> bool:
    return any(file_contains_tag(filename, tag=tag) for filename in filenames)
