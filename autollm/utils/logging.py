# autollm/utils/logging.py

import logging

# Configure the logger
logger = logging.getLogger('autollm')
logger.setLevel(logging.INFO)

# Create a console handler with a specific format
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

# Ensure that log messages propagate up to the root logger
logger.propagate = False
