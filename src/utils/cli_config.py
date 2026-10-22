import sys


def explicit_cli_dests(parser, argv=None):
    """Return argparse destinations explicitly present in argv."""
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit = set()
    stop_options = False
    for token in argv:
        if stop_options:
            continue
        if token == "--":
            stop_options = True
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(dest)
    return explicit


def merge_config_with_cli(parser, parsed_args, config, argv=None):
    """Merge YAML config with argparse values.

    Precedence:
    1. argparse defaults provide keys missing from YAML;
    2. YAML overrides argparse defaults;
    3. arguments explicitly present on the CLI override YAML.
    """
    cli_values = vars(parsed_args)
    merged = dict(config or {})
    explicit = explicit_cli_dests(parser, argv)
    for key, value in cli_values.items():
        if key not in merged or key in explicit:
            merged[key] = value
    return merged
