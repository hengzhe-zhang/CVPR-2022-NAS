import atexit
import functools

import requests


def send_message(message):
    url = f'http://127.0.0.1:5010/notify?message={message}'
    requests.get(url)


def exit_handler():
    send_message('Experiment has terminated.')


def notify(func=None, name='ML'):
    if func is None:
        return functools.partial(notify, name=name)

    @functools.wraps(func)
    def with_logging(*args, **kwargs):
        atexit.register(exit_handler)
        send_message(f'{name} experiment start success!')
        try:
            result = func(*args, **kwargs)
            send_message(f'{name} experiment is finished.')
            return result
        except Exception as e:
            # traceback.print_exc()
            send_message(f'{name} experiment has encountered some errors.')
            raise e

    return with_logging


if __name__ == '__main__':
    send_message('test')
