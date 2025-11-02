import logging
import os
from datetime import datetime


def configure_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("teaching_agent")


logger = configure_logging()
