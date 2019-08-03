class ParamController:

    def __init__(self, initial_value=0.01):

        self._value = initial_value
        self._epoch = 0

    def set_epoch(self, epoch):

        self._epoch = epoch

    def get_value(self):

        import os
        if os.environ.get('reg_const') is not None:
            return self._value

        if self._epoch <= 20:
            return self._value
        elif self._epoch <= 60:
            return self._value * 1e-3

        return self._value


class HtriParamController:

    def __init__(self, initial_value=1.):

        self._value = initial_value
        self._epoch = 0

    def set_epoch(self, epoch):

        self._epoch = epoch

    def get_value(self):

        import os

        try:
            decay_to = float(os.environ.get('htri_decay'))
        except (TypeError, ValueError):
            return self._value

        if self._epoch > 100:
            return decay_to
        else:
            return self._value + (decay_to - self._value) * (self._epoch / 100)
