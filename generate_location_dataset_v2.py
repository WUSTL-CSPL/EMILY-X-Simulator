#!/usr/bin/env python3
"""Compatibility entry point. All generation logic lives in generate_location_dataset.py.

New commands should use that module directly with --profile near_south10.
"""
from generate_location_dataset import main


if __name__ == "__main__":
    main(default_profile="near_south10", default_count=2000,
         default_incident_target=1000, default_workers=8)
