"""Shared building blocks for the multislice and 3-D volume degradations.

Layout ``(B, C, D, H, W)``: D = slice (multislice) or kz (3-D), H = phase-encode, W = readout.

Host syncs: small constants are built on the device (:func:`volume_warp.device_constants`), and
a CPU-built static schedule (line times, masks, state indices) is uploaded at most once per call;
a data-dependent device reduction read back on the host is never allowed.
"""
