"""
Module providing data minimization for ML.

This module implements a first-of-a-kind method to help reduce the amount of personal data needed to perform
predictions with a machine learning model, by removing or generalizing some of the input features. For more information
about the method see: http://export.arxiv.org/pdf/2008.04113

The main class, ``GeneralizeToRepresentative``, is a scikit-learn compatible ``Transformer``, that receives an existing
estimator and labeled training data, and learns the generalizations that can be applied to any newly collected data for
analysis by the original model. The ``fit()`` method learns the generalizations and the ``transform()`` method applies
them to new data.

It is also possible to export the generalizations as feature ranges.

"""
from apt.minimization.minimizer import GeneralizeToRepresentative
from apt.minimization.security_metrics import compute_pa_ilag_score, measure_attribute_disclosure
from apt.minimization.security_postprocess import enforce_cell_privacy, randomize_cell_representatives
from apt.minimization.weighted_ncp import compute_sensitivity_weights
