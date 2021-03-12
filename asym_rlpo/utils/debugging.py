from typing import Type


def checkraise(
    condition: bool,
    error_type: Type[Exception],
    error_message_fmt: str,
    *args,
    **kwargs
):
    if not condition:
        raise error_type(error_message_fmt.format(*args, **kwargs))
