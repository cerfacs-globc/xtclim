"""
Train file to launch pipeline
"""

import os
import sys
from typing import Dict
import argparse
import logging
from datetime import datetime


from itwinai.parser import ConfigParser, ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        required=True,
        help="Configuration file to the pipeline to execute.",
    )
    args = parser.parse_args()

    pipe_parser = ConfigParser(
        config=args.pipeline,
    )

    pipeline = pipe_parser.parse_pipeline()
    pipeline.execute()
