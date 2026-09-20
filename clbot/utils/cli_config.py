"""Module to parse arguments from CLI"""

from argparse import ArgumentParser, Namespace


def arg_parser() -> Namespace:
    """Function to parse arguments

    Returns
    -------
        Namespace: populated namespace from arguments

    """
    parser = ArgumentParser(description="Run clbot from CLI")
    parser.add_argument(
        "--start",
        "-s",
        dest="start",
        action="store_true",
        help="Start the bot when the program opens",
    )
    parser.add_argument("--config", dest="config", default="config.yaml", help="Path to central config file")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-humanize", dest="no_humanize", action="store_true", help="Disable humanized clicks/pauses")
    parser.add_argument("--max-battles", dest="max_battles", type=int, default=0, help="Stop after N battles (0=unlimited)")
    return parser.parse_args()
