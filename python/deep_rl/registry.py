from .common import train_wrappers as wrappers
from .core import AbstractAgent

_registry = dict()
_agent_registry = dict()


def get_dynamic_name():
    import inspect
    import os
    frm = inspect.stack()[2]
    mod = inspect.getmodule(frm[0])

    _, name = os.path.split(mod.__file__)
    name = os.path.splitext(name)[0]
    return name.replace('_', '-')


def register_agent(id=None, **kwargs):
    if id is None:
        id = get_dynamic_name()

    print('Registering agent %s' % id)

    def wrap(agent):
        _agent_registry[id] = dict(agent=agent, **kwargs)
        return agent

    return wrap


def register_trainer(id=None, **kwargs):
    if id is None:
        id = get_dynamic_name()

    print('Registering trainer %s' % id)

    def wrap(trainer):
        _registry[id] = dict(trainer=trainer, **kwargs)
        return trainer
    return wrap


def make_trainer(id=None, **kwargs):
    if id is None:
        id = get_dynamic_name()

    wargs = dict(**_registry[id])
    del wargs['trainer']

    tkwargs = dict(**wargs)
    tkwargs.update(kwargs)
    instance = _registry[id]['trainer'](name=id, **tkwargs)
    instance = wrappers.wrap(instance, **wargs).compile()
    return instance


def make_agent(id, **kwargs):
    if isinstance(id, str):
        wargs = dict(**_agent_registry[id])
        del wargs['agent']

        wargs.update(kwargs)
        instance = _agent_registry[id]['agent'](name=id, **wargs)
        return instance
    else:
        return id(**kwargs)
