"""
Some parts adapted from Magic UV:
    https://github.com/nutti/Magic-UV/blob/master/uv_magic_uv/common.py
"""
import bpy
import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    logfile = '/tmp/blender_render.log'
    open(logfile, 'a').close()

    # redirect output to log file
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    yield

    os.close(1)
    os.dup(old)
    os.close(old)


def check_version(major, minor, _):
    """
    Check blender version
    """

    if bpy.app.version[0] == major and bpy.app.version[1] == minor:
        return 0
    if bpy.app.version[0] > major:
        return 1
    if bpy.app.version[1] > minor:
        return 1
    return -1
