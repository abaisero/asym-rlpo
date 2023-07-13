def int_gt(threshold: int):
    def inner(x: str) -> int:
        x_int = int(x)

        if x_int <= threshold:
            raise ValueError(f'Argument should be greater than {threshold}')

        return x_int

    return inner


def int_ge(threshold: int):
    def inner(x: str) -> int:
        x_int = int(x)

        if x_int < threshold:
            raise ValueError(
                f'Argument should be greater-or-equal than {threshold}'
            )

        return x_int

    return inner


int_pos = int_gt(0)
int_non_neg = int_ge(0)


def int_pow_2(x: str) -> int:
    x_int = int(x)

    if x_int <= 0 or x_int & (x_int - 1) != 0:
        raise ValueError('Argument should be power of two')

    return x_int
