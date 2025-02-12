"""
Train file to launch pipeline
"""

from itwinai.parser import ArgumentParser, ConfigParser

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
