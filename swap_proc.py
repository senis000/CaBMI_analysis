import os


def swapping_proc(pathname, proc, *args, **kwargs):
    os.system("swapon {0}".format(pathname))
    proc(*args, **kwargs)
    os.system("swapoff -a")