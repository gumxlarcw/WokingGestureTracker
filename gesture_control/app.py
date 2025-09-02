from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.FATAL)

from controller import run_loop

if __name__ == "__main__":
    run_loop()
