def bool_arg(arg):
    assert arg.lower() in ('true', 'false'), f'{arg} is not a valid value for a boolean argument'
    return arg.lower() == 'true'
