import sys

from . import api
from .parser import get_main_parser

__all__ = ['main']


def get_run_args(parser_fn=get_main_parser, printed=True):
    parser = parser_fn()
    if len(sys.argv) > 1:
        args = parser.parse_args()
        if printed:
            param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
            print('usage: %s\n%s\n%s\n' % (' '.join(sys.argv), '_' * 50, param_str))
        return args
    else:
        parser.print_help()
        exit()


def main():
    args = get_run_args()
    getattr(api, args.cli, no_cli_error)(args)


def no_cli_error(*args, **kwargs):
    get_main_parser().print_help()
    exit()
    # pass