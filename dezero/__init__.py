is_simple_core = True

from dezero.core_simple import Variable, Function

if is_simple_core:
    from dezero.core_simple import (
        using_config,
        no_grad,
        as_array,
        as_variable,
        setup_variable,
    )


setup_variable()
