"""Prior-method baselines for the ``kspace_filling`` cohort, kept out of the framework tree.

Each module wraps one vendored upstream (``external/baselines/<name>/``) behind the
in-tree :class:`spectramr.models.baselines.BaselineAdapter` contract, and
:mod:`spectramr_baselines.strategy` trains it through the authors' own forward process.
Importing a module registers its model; nothing is imported here so that one missing
upstream does not disable the other two.
"""
