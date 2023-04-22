"""
CRITICAL 50
ERROR 40
WARNING 30
INFO 20
DEBUG 10
NOTSET 0
"""
import logging
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING").upper())
log = logging.getLogger("")
