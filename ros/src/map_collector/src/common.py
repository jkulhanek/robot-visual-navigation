class Proxy(object):
    def __init__(self, inner):
        object.__setattr__(
            self,
            "_obj",
            inner
        )

    #
    # proxying (special cases)
    #
    def __getattribute__(self, name):
        value = getattr(object.__getattribute__(self, "_obj"), name)
        if callable(value):
            fn = value.__func__
            value = lambda *args,**kwargs: fn(self, *args, **kwargs)
        return value

    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)

    def __nonzero__(self):
        return bool(object.__getattribute__(self, "_obj"))

    def __str__(self):
        return str(object.__getattribute__(self, "_obj"))

    def __repr__(self):
        return repr(object.__getattribute__(self, "_obj"))

    def __hash__(self):
        return hash(object.__getattribute__(self, "_obj"))