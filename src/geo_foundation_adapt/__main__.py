import os

# Set before torch is imported so OpenMP honors the CPU budget.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from geo_foundation_adapt.cli import main

raise SystemExit(main())
