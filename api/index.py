"""Vercel entry point: serves the Max AI Flask app as a Python serverless function."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Max.AI"))

from MAX_AI import app  # noqa: E402,F401  (Vercel looks for a WSGI app named `app`)
