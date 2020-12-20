# -*- coding: utf-8 -*-
"""
    timethis
    ~~~~~

    A quick and dirty profiling tool.

    :copyright: (c) 2010 by Elben Shira.
    :license: MIT, see LICENSE
"""

import time


def timethis(target=None, persist=False, outfile=None, display=True):
    # TODO: implement write to outfile and display (write to stdout)
    #       refactor out the print statements

    if hasattr(target, '__call__'):
        # 'target' is a function
        def decorated_func(*args, **kwargs):
            start = time.time()
            r = target(*args, **kwargs)
            total = time.time() - start

            name, filename, linenum = info(target)
            print("%s:%d (%s) %.06f seconds" % (filename, linenum, name, total))

            return r

        return decorated_func
    # elif type(target) == str and target == 'persist':
    elif target is None:
        # assume we want persistence
        # user passed in options
        if persist:
            runs = []

        def decorator(target):
            def decorated_func(*args, **kwargs):
                start = time.time()
                r = target(*args, **kwargs)
                total = time.time() - start
                if persist:
                    runs.append(total)

                # TODO: print number of runs we've done
                name, filename, linenum = info(target)

                if persist:
                    print("%s:%d (%s) %.06f seconds (avg: %.06f over %d runs)" % (filename,
                                                                            linenum, name, total, sum(runs) / len(runs),
                                                                            len(runs)))
                else:
                    print("%s:%d (%s) %.06f seconds" % (filename, linenum, name, total))

                return r

            return decorated_func

        return decorator


def info(target):
    """
    Get basic code information about 'target'.
    """
    co = target.__code__
    return (co.co_name, co.co_filename, co.co_firstlineno)
