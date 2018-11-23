"""
===================================================
General code tools (:mod:`fklab.codetools.general`)
===================================================

.. currentmodule:: fklab.codetools.general

Tools for marking functions as deprecated.

.. autosummary::
    :toctree: generated/
    
    deprecated

"""

import functools
import inspect
import warnings

__all__ = ['deprecated']

# deprecated decorator was borred from https://gist.github.com/kgriffs/8202106

# We don't want our deprecations to be ignored by default,
# so create our own type.
class FKLabDeprecationWarning(UserWarning):
    pass


def deprecated(instructions="Please avoid use in the future."):
    """Create decorator that flags a method as deprecated.
    
    Parameters
    ----------
    instructions : str
        A human-friendly string of instructions, e.g.
        'Please migrate to add_proxy() ASAP.'
    
    """
    
    def decorator(func):
        '''This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.'''
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = 'Call to deprecated function {}. {}'.format(
                func.__name__,
                instructions)

            frame = inspect.currentframe().f_back

            warnings.warn_explicit(message,
                                   category=FKLabDeprecationWarning,
                                   filename=inspect.getfile(frame.f_code),
                                   lineno=frame.f_lineno)

            return func(*args, **kwargs)

        return wrapper

    return decorator
