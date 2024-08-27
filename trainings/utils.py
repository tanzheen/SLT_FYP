import numpy as np 
import torch 
import torch.distributed as dist 
import torch.nn.functional as F 
import time 
import datetime 
from collections import defaultdict, deque 
from definition import * 
import random

class SmoothedValue(object): # Actually pytorch also has a SmoothedValue function but we shall just go with this for now
    """
    Tracks a series of values and provides access to smoothed values over a
    specified window size or the global series average.

    Attributes:
        deque (collections.deque): A deque to store the most recent values up to the specified window size.
        total (float): The cumulative total of all values added.
        count (int): The total number of values added.
        fmt (str): A format string for output, default is "{median:.4f} ({global_avg:.4f})".
    """

    def __init__(self, window_size=20, fmt=None):
        """
        Initializes the SmoothedValue object with a specified window size and format.

        Args:
            window_size (int): The number of recent values to consider for smoothing. Default is 20.
            fmt (str): A format string for displaying the smoothed values. Default is "{median:.4f} ({global_avg:.4f})".
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Updates the deque with a new value and increments the total and count.

        Args:
            value (float): The value to add to the series.
            n (int): The number of occurrences of the value. Default is 1.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n 

    def synchronize_between_processes(self):
        """
        Synchronizes the count and total values across all processes in a distributed environment.

        Warning:
            This method does not synchronize the deque, only the count and total values.

        Note:
            This method assumes that distributed processing is being used with PyTorch.
            However, currently, distributed processing is not being used
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """
        Calculates the median of the values in the deque.

        Returns:
            float: The median of the stored values.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        Calculates the average of the values in the deque.

        Returns:
            float: The average of the stored values.
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """
        Calculates the global average of all the values added (total / count).

        Returns:
            float: The global average of the series.
        """
        return self.total / self.count

    @property
    def max(self):
        """
        Retrieves the maximum value from the deque.

        Returns:
            float: The maximum value in the stored values.
        """
        return max(self.deque)

    @property
    def value(self):
        """
        Retrieves the most recent value added to the deque.

        Returns:
            float: The latest value in the series.
        """
        return self.deque[-1]

    def __str__(self):
        """
        Formats the smoothed values as a string using the provided format.

        Returns:
            str: A formatted string of the median and global average.
        """
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )
    
# Need to go and do more research on how KL loss works
class KLLoss(torch.nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=torch.nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

import time
import datetime
from collections import defaultdict
import torch

class MetricLogger(object):
    """
    A logger that tracks various metrics during training and provides utilities 
    for synchronized logging across processes and formatted output.
    
    Attributes:
        meters (defaultdict): A dictionary that maps metric names to SmoothedValue objects.
        delimiter (str): A string delimiter used for separating logged outputs.
    """

    def __init__(self, delimiter="\t"):
        """
        Initializes the MetricLogger with a specified delimiter.

        Args:
            delimiter (str): The delimiter to use when printing the metrics. Default is tab ("\t").
        """
        self.meters = defaultdict(SmoothedValue) # defaultdict hold instances of the SmoothedValue calss for each metric being tracked
        self.delimiter = delimiter # String used to separate the different metrics when they are printed

    def update(self, **kwargs):
        """
        Updates the stored metrics with new values. Each metric is updated based on its name.

        Args:
            **kwargs: Keyword arguments where each key is the metric name and the value is the metric value.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()  # Convert Tensor to a Python float/int if needed by using .item()
            assert isinstance(v, (float, int)), "Metric values must be float or int."
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        Custom attribute access method. Retrieves a meter by name if it exists.

        Args:
            attr (str): The name of the attribute or meter to retrieve.

        Returns:
            SmoothedValue: The meter corresponding to the attribute name.
        
        Raises:
            AttributeError: If the attribute or meter does not exist.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """
        Returns a string representation of the current metrics.

        Returns:
            str: A formatted string of all current metrics.
        """
        loss_str = []
        # loop to print all the metrics stored
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        Synchronizes all meters across processes in a distributed environment.
        Useful when running training across multiple GPUs. 
        Currently not used at all
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        Adds a new meter to the logger with a specified name.

        Args:
            name (str): The name of the metric.
            meter (SmoothedValue): An instance of SmoothedValue to track the metric.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Logs metrics at a specified frequency during iteration over an iterable.
        Also estimates the remaining time and logs GPU memory usage if available.

        Args:
            iterable (iterable): The iterable over which to iterate, typically the data loader.
            print_freq (int): How often to print the log (every `print_freq` iterations).
            header (str): A string header to prepend to the logs (optional).

        Yields:
            The next item from the iterable.
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0 # convert GPU memory usage from bytes to megabytes for readability 
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# count number of parameters in model 
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6



def data_augmentation(resize=(320, 240), crop_size=224, is_train=True):
    if is_train:
        left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
    else:
        left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2

    return (left, top, left + crop_size, top + crop_size), resize


def NoiseInjecting(raw_gloss, noise_rate=0.15, noise_type='omit_last', random_shuffle=False, is_train=True):
    new_gloss = []

    for ii, gloss in enumerate(raw_gloss):
        text = gloss.split()

        if noise_type == 'omit':
            # del noise
            if random.uniform(0, 1) <= 1. and is_train:
                index = sampler_func(len(text), int(len(text)*(1. - noise_rate)), random_choice=is_train)
                noise_gloss = []
                noise_idx = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
                        noise_idx.append(i)
            else:
                noise_gloss = [d for d in text]

        elif noise_type == 'omit_last' :
            if random.uniform(0, 1) <= 1.0 and is_train:
                index = np.arange(0, len(text) - int(np.ceil(len(text)*(np.random.uniform(0,noise_rate,(1,))))), 1, dtype=int)
                noise_gloss = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
            else:
                noise_gloss = [d for d in text]
        
        if is_train and random_shuffle and random.uniform(0, 1) > 0.5:
            random.shuffle(noise_gloss) # random shuffle sequence

        new_gloss.append(' '.join(noise_gloss))
    return new_gloss



def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                        for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                        for i in range(sn)]
    return f(clip)
