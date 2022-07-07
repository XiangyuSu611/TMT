import os
from pathlib import Path

import contextlib

toolbox_home = Path(os.path.expanduser(os.getenv('K_TOOLBOX_HOME', '~/.toolbox')))


def exists(name):
    return get_path(name).exists()


def get_path(name):
    ask_make_path()
    return toolbox_home / name


def ask_make_path():
    if not toolbox_home.exists():
        if input(f"{toolbox_home} does not exist, create? (y/n) ") == 'y':
            toolbox_home.mkdir(parents=True)


@contextlib.contextmanager
def open_file(name, mode='r'):
    with open(toolbox_home / name, mode) as f:
        yield f
