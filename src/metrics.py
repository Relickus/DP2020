from collections import deque
import numpy as np

class Metric(object):
    def __init__(self, name, title=None):
        self._name = name
        self._title = title
    def __str__(self):
        return f'{self._name}:{self._func():.4f}'
    def __repr__(self):
        return self.__str__()

    @property
    def title(self):
        if self._title is None:
            return ""
        return self._title

    def item(self):
        return self._func()

    def update(self,val):
        self._func(val)

class AVG(Metric):
    def __init__(self, name="Avg", title=None):
        super().__init__(name=name, title=title)
        self._func = self.avg
        self._n_vals = 0
        self._sum_sofar = 0
        self._mean = 0

    def avg(self, val=None):
        if val:
            self._sum_sofar += val
            self._n_vals +=1
            self._mean = self._sum_sofar / self._n_vals
        return self._mean


class MA(Metric):
    def __init__(self, steps, name="MA",title=None):
        super().__init__(name=name+str(steps), title=title)
        self._func = self.ma(steps)

    def ma(self,steps):
        """
        Moving average metric.
        """
        q=deque()
        def ma_func(val=None):
            if val: q.append(val)
            while len(q) > steps:
                q.popleft()
            return np.mean(q)
        return ma_func
