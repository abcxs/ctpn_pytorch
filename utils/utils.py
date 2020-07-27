from contextlib import contextmanager
import datetime
import time
import random
import uuid

import numpy as np
import torch

from .logger_helper import get_root_logger

logger = get_root_logger()

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_checkpoint(model, file_name, only_weights=False, map_location=None):
    if file_name:
        checkpoint = torch.load(file_name, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if list(state_dict.keys())[0].startswith("module."):
            # module.module.cnn.conv0.weight module.cnn.conv0.weight
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if only_weights:
            return {}
        return checkpoint
    return {}


def time2hms(eta):
    h = eta // 3600
    eta -= h * 3600
    m = eta // 60
    s = eta - m * 60
    return h, m, s


def save_checkpoint(file_name, model, optimerzer=None, scheduler=None, meta=None):
    checkpoint = {}
    checkpoint["state_dict"] = model.state_dict()
    if optimerzer:
        checkpoint["optimizer"] = optimerzer.state_dict()
    if scheduler:
        checkpoint["scheduler"] = scheduler.state_dict()
    if meta:
        checkpoint["meta"] = meta
    torch.save(checkpoint, file_name)


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
        tuple(list): A tuple containing multiple list, each list contains
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret

def generate_request_id():
    return "{0}_{1}".format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"), str(uuid.uuid4())[:8])

class TimeRecord(object):
    def __init__(self, time_key):
        self.start_time = time.time()
        self.end_time = -1
        self.time_elapse = -1
        self.time_key = time_key
        logger.info("{0} starts".format(self.time_key))

    def stop(self):
        self.end_time = time.time()
        self.time_elapse = self.end_time - self.start_time
        logger.info("{0} ends, time elapse: {1}".format(self.time_key, self.time_elapse))


@contextmanager
def time_record(time_key):
    record = TimeRecord(time_key)
    yield record
    record.stop()


def enable_time_record(func, time_key=None):
    time_key = time_key if time_key else func.__name__

    def wrapper(*args, **kwargs):
        record = TimeRecord(time_key)
        func_ret = func(*args, **kwargs)
        record.stop()
        return func_ret
    return wrapper