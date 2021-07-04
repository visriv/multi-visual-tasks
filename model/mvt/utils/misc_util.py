import functools
import itertools
import subprocess
import warnings
import torch
from torch import distributed as dist
from collections import abc
from importlib import import_module
from functools import partial
import time
import os
import sys
from shutil import get_terminal_size
from collections.abc import Iterable

from .log_util import get_root_logger

DEBUG_COMPLETED_TIME = bool(os.environ.get('DEBUG_COMPLETED_TIME', False))
logger = get_root_logger()


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_str(x):
    """Whether the input is an string instance.
    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.
    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.
    Returns:
        list[module] | module | None: The imported modules.
    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def iter_cast(inputs, dst_type, return_type=None):
    """Cast elements of an iterable object into some type.
    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be
            converted to this type, otherwise an iterator.
    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    """Cast elements of an iterable object into a list of some type.
    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    """Cast elements of an iterable object into a tuple of some type.
    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.
    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.
    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length.
    Args:
        in_list (list): The list to be sliced.
        lens(int or list): The expected length of each out list.
    Returns:
        list: A list of sliced list.
    """
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError('"indices" must be an integer or a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError('sum of lens and list length does not '
                         f'match: {sum(lens)} != {len(in_list)}')
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    """Concatenate a list of list into a single list.
    Args:
        in_list (list): The list of list to be merged.
    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))


def check_prerequisites(
        prerequisites,
        checker,
        msg_tmpl='Prerequisites "{}" are required in method "{}" but not '
        'found, please install them first.'):  # yapf: disable
    """A decorator factory to check if prerequisites are satisfied.
    Args:
        prerequisites (str of list[str]): Prerequisites to be checked.
        checker (callable): The checker method that returns True if a
            prerequisite is meet, False otherwise.
        msg_tmpl (str): The message template with two variables.
    Returns:
        decorator: A specific decorator.
    """

    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(
                prerequisites, str) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def _check_py_package(package):
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd):
    if subprocess.call(f'which {cmd}', shell=True) != 0:
        return False
    else:
        return True


def requires_package(prerequisites):
    """A decorator to check if some python packages are installed.
    Example:
        >>> @requires_package('numpy')
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        array([0.])
        >>> @requires_package(['numpy', 'non_package'])
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        ImportError
    """
    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    """A decorator to check if some executable files are installed.
    Example:
        >>> @requires_executable('ffmpeg')
        >>> func(arg1, args):
        >>>     print(1)
        1
    """
    return check_prerequisites(prerequisites, checker=_check_executable)


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
        
    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


class NiceRepr(object):
    """Inherit from this class and define ``__nice__`` to "nicely" print your
    objects.

    Defines ``__str__`` and ``__repr__`` in terms of ``__nice__`` function
    Classes that inherit from :class:`NiceRepr` should redefine ``__nice__``.
    If the inheriting class has a ``__len__``, method then the default
    ``__nice__`` method will return its length.
    Example:
        >>> class Foo(NiceRepr):
        ...    def __nice__(self):
        ...        return 'info'
        >>> foo = Foo()
        >>> assert str(foo) == '<Foo(info)>'
        >>> assert repr(foo).startswith('<Foo(info) at ')
    Example:
        >>> class Bar(NiceRepr):
        ...    pass
        >>> bar = Bar()
        >>> import pytest
        >>> with pytest.warns(None) as record:
        >>>     assert 'object at' in str(bar)
        >>>     assert 'object at' in repr(bar)
    Example:
        >>> class Baz(NiceRepr):
        ...    def __len__(self):
        ...        return 5
        >>> baz = Baz()
        >>> assert str(baz) == '<Baz(5)>'
    """

    def __nice__(self):
        """str: a "nice" summary string describing this module"""
        if hasattr(self, '__len__'):
            # It is a common pattern for objects to use __len__ in __nice__
            # As a convenience we define a default __nice__ for these objects
            return str(len(self))
        else:
            # In all other cases force the subclass to overload __nice__
            raise NotImplementedError(
                f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        """str: the string of the module"""
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        """str: the string of the module"""
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)


class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer:
    """A flexible Timer class."""

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time.time()
            self._is_running = True
        self._t_last = time.time()

    def since_start(self):
        """Total time since the timer is started.
        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time.time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.
        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.
        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time.time() - self._t_last
        self._t_last = time.time()
        return dur


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            'elapsed: 0s, ETA:')
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        self.file.flush()


def track_progress(func, tasks, bar_width=50, file=sys.stdout, **kwargs):
    """Track the progress of tasks execution with a progress bar.
    Tasks are done with a simple for-loop.
    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.
    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    prog_bar.file.write('\n')
    return results


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float, optional): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs
