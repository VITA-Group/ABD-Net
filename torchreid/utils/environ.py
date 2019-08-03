import os


def get_env_or_raise(checker, *names):

    for name in names:
        try:
            value = checker(os.environ.get(name))
        except Exception as e:
            print(e)
        else:
            return value

    raise RuntimeError('No suitable envvar found in names {}'.format(names))
