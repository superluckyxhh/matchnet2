def dynamic_load(config_name):
    config_path = f'configs.{config_name}'
    mod = __import__(config_path, fromlist=[''])
    return mod.get_args_parser()