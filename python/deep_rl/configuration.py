def _merge_value(original_value, value):
    if original_value is None or value is None:
        return value

    if isinstance(original_value, (dict, Configuration)):
        for key, val in value.items():
            original_value[key] = _merge_value(
                original_value.get(key, None), val)

        return original_value

    return value


class Configuration:
    def __init__(self):
        self._store = dict()
        self.visdom = None

    def _merge_value(self, original_value, value):
        if original_value is None or value is None:
            return value

        if isinstance(original_value, (dict, Configuration)):
            for key, val in value.items():
                original_value[key] = self._merge_value(
                    original_value.get(key, None), val)

            return original_value

        return value

    def __getitem__(self, key):
        return self._store.__getitem__(key)

    def __setitem__(self, key, val):
        self._store.__setitem__(key, val)

    def __delitem__(self, key):
        self._store.__delitem__(key)

    def __contains__(self, key):
        return self._store.__contains__(key)

    def update(self, config):
        return self._merge_value(self, config)

    def get(self, path='', default=None):
        frags = path.split('.')

        value = self._store
        for frag in frags:
            if not frag in value:
                return default

            value = value[frag]

        return value


def wrap_config(config):
    if config is None:
        return config

    if isinstance(config, (dict)):
        return ConfigWrapper(config)

    return config


_config_attrs = ['get', '__getattribute__', '__setattr__',
                 '__str__', '__repr__', '__iter__', 'data', 'as_dict', 'update']


class ConfigWrapper:
    def __init__(self, data):
        super().__setattr__('data', data)

    def __getattribute__(self, key):
        if key in _config_attrs:
            return super().__getattribute__(key)

        return wrap_config(self.data.get(key, None))

    def __setattr__(self, key, value):
        if key in _config_attrs:
            super().__setattr__(key, value)

        self.data[key] = _merge_value(self.data.get(key, None), value)

    def __delattr__(self, key):
        if key in _config_attrs:
            super().__delattr__(key)

        del self.data[key]

    def get(self, path='', default=None):
        frags = [x for x in path.split('.') if x != '']

        value = self.data
        for frag in frags:
            if not frag in value:
                return wrap_config(default)

            value = value[frag]
        return wrap_config(value)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return "<ConfigurationWrapper %s>" % repr(self.data)

    def __iter__(self):
        return self.data.__iter__()

    def as_dict(self):
        return self.data

    def update(self, other):
        for key, val in other.items():
            setattr(self, key, val)


store = dict()
configuration = wrap_config(store)

# Apply default configuration
configuration.update(dict(
    models_path='./checkpoints'
))
