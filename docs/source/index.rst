.. canml documentation master file, created by
   sphinx-quickstart on Sun Apr 27 13:50:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to canml’s documentation!
=================================
.. image:: _static/canml-icon.svg
   :alt: canml icon
   :align: right
   :height: 2em
   
canml
-----
Modern toolkit for end-to-end CAN bus log processing—whether you’re streaming huge BLF files in chunks or doing a full in-memory decode—built on top of pandas, cantools, and python-can.

Features at a glance
--------------------

**Configurable BLF loading via CanmlConfig** (chunk size, progress bars, uniform timing, interpolation, sorting)

**DBC management with caching**: merge one or many .dbc files, auto-detect collisions, optional signal-name prefixing, and LRU caching for repeated loads

**Streaming BLF decode** with iter_blf_chunks()—filter by arbitration ID or by signal name, and emit tidy pandas chunks without blowing up memory

**One-call full-file load** with load_blf()—ID or signal filters, uniform timestamp spacing (with raw backup), missing-signal injection (dtype-safe), linear interpolation, and automatic enum→Categorical conversion

**Rich metadata support**: pull any custom DBC attributes into df.attrs['signal_attributes'] and carry them through to exports

**Incremental CSV & Parquet export**: auto-create directories, write in append mode, and side-dump your signal_attributes JSON alongside your data

**Built-in logging & progress bars** (Python logging + tqdm) to keep you informed without clutter


Whether you need to slice-and-dice hundreds of gigabytes of CAN traffic or just spin up a few quick plots, canml handles the heavy lifting so you can focus on insights.


Installation
------------
.. code-block:: console

    pip install canml

Quickstart
----------
.. code-block:: python

   from canml.canmlio import load_dbc_files, load_blf, to_csv, to_parquet, CanmlConfig

   # 1️⃣ Load your DBC(s) (namespace-collision safe)
   #    If you have multiple, pass a list; prefix_signals avoids any name clashes.
   db = load_dbc_files("vehicle.dbc", prefix_signals=True)

   # 2️⃣ Stream-decode the BLF in memory (no CSV yet)
   #    Use CanmlConfig to tweak chunk size, uniform-timing, interpolation, etc.
   cfg = CanmlConfig(
      chunk_size=5000,        # rows per chunk
      force_uniform_timing=True,
      interval_seconds=0.01,
      progress_bar=True
   )
   df = load_blf(
      blf_path="drive.blf",
      db=db,
      config=cfg,
      expected_signals=["Engine_RPM", "Brake_Active"]
   )

   # 3️⃣ Inspect your DataFrame
   print(df.head())
   #     timestamp  Engine_RPM  Brake_Active  raw_timestamp
   # 0        0.00       1200.0           0.0       162523.1
   # 1        0.01       1230.0           0.0       162523.2
   # …

   # 4️⃣ Export to CSV (with metadata JSON)
   to_csv(
      df,
      output_path="drive_data.csv",
      metadata_path="drive_data_signals.json"
   )

   # 5️⃣ Or write Parquet (faster reads/writes + metadata)
   to_parquet(
      df,
      output_path="drive_data.parquet",
      metadata_path="drive_data_signals.json"
   )


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


