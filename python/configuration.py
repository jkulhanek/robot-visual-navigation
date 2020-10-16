import os
import json

default_configuration = dict(
    # visdom = dict(
    #    server = 'http://localhost',
    #    port = 8097
    # ),
    models_path='~/models',
    videos_path='~/results/videos'
)

basepath = os.path.expanduser('~/.visual_navigation')
os.makedirs(basepath, exist_ok=True)
configuration = dict(**default_configuration)
if not os.path.exists(os.path.join(basepath, 'config')):
    with open(os.path.join(basepath, 'config'), 'w+') as f:
        json.dump(configuration, f)

with open(os.path.join(basepath, 'config'), 'r') as f:
    configuration.update(**json.load(f))


def expand_user(d):
    if isinstance(d, dict):
        dnew = dict()
        for key, v in d.items():
            if key.endswith('_path') and isinstance(v, str) and v.startswith('~'):
                dnew[key] = os.path.expanduser(v)
            else:
                dnew[key] = expand_user(v)
        return dnew

    return d


configuration = expand_user(configuration)
