.. canml documentation master file, created by
   sphinx-quickstart on Sun Apr 27 13:50:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to canml’s documentation!
=================================

canml is a toolkit for streamlined in-memory and on-disk processing of CAN bus logs in BLF format, leveraging pandas and cantools.

Features at a glance
--------------------
- **DBC management:** Merge one or more DBC files with optional signal-name prefixing.
- **Streaming BLF decoding:** Efficient chunked reads for large logs via `iter_blf_chunks()`.
- **Full-file BLF loading:** One-call load with filtering, uniform timing, and missing-signal injection.
- **Incremental CSV & Parquet export:** Write large datasets without exhausting memory.
- **Built-in logging & progress bars** via Python logging and tqdm.

Installation
------------
.. code-block:: console

    pip install canml

Quickstart
----------
.. code-block:: python

    from canml.canmlio import load_dbc_files, load_blf, to_csv, to_parquet

    # Merge DBCs
    db = load_dbc_files(['veh1.dbc', 'veh2.dbc'], prefix_signals=True)

    # Load BLF file
    df = load_blf('log.blf', db, force_uniform_timing=True, interval_seconds=0.01)

    # Export
    to_csv(df, 'out.csv')
    to_parquet(df, 'out.parquet')

Contents
--------
.. toctree::
   :maxdepth: 2
   :caption: API Reference

   canml
   canml.canmlio

.. toctree::
   :maxdepth: 1
   :caption: Guides

   tutorials/streaming
   tutorials/full_load
   tutorials/export

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


