"""The one owner of checkpoint files: where they live, what they hold, how they load.

Deliberately empty: ``layout``, ``select`` and ``ref`` are imported by ``cli/`` and must
not pull torch, which the codecs import.
"""
