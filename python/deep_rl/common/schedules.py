from ..core import Schedule


class LinearSchedule(Schedule):
    def __init__(self, start, end, total_iterations):
        self.start = start
        self.end = end
        self.total_iterations = total_iterations
        self.time = None

    def __call__(self):
        return self.end + (0 if self.time >= self.total_iterations else float(self.start - self.end) * float(self.total_iterations - self.time) / float(self.total_iterations))


class ConstantSchedule(Schedule):
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class MultistepSchedule(Schedule):
    def _wrap_schedule(self, schedule):
        if isinstance(schedule, Schedule):
            return schedule

        return ConstantSchedule(schedule)

    def __init__(self, initial_value, steps):
        self.initial_value = self._wrap_schedule(initial_value)
        self.steps = [(i, self._wrap_schedule(x)) for i, x in steps]
        self.time = None
        self.current_schedule = None

    def __call__(self):
        return self.current_schedule()

    def _set_current_schedule(self):
        val = self.initial_value
        time = 0
        for i, value in self.steps:
            if self.time >= i:
                val = value
                time = i
        self.current_schedule = val
        return time

    def step(self, time):
        super().step(time)
        start_time = self._set_current_schedule()
        self.current_schedule.step(self.time - start_time)
