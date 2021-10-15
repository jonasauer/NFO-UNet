
class AbstractConfig(object):

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict=None):
        if new_config_dict is None:
            new_config_dict = {}
        ret = AbstractConfig(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        if isinstance(new_config_dict, AbstractConfig):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def __str__(self):
        data = '\nConfig {\n'
        for k, v in vars(self).items():
            v_str = f'{v}'.replace(' ', '').replace('\n', ' ')
            print_msg = f'{k} = {v_str}'
            data = f'{data}  {print_msg}\n'
        return f'{data}{"}"}'
