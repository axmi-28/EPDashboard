"""EPDashboard — feature-dashboard tooling for Exemplar Partitioning dictionaries.

Region-level dashboards in the spirit of SAEDashboard's feature dashboards
(github.com/jbloomAus/SAEDashboard), with the EP-native replacements:

    SAEDashboard (feature)          EPDashboard (region)
    ----------------------          --------------------
    top activating examples         closest members (distance to exemplar)
    quantile-interval samples       members across distance bands + random draw
    per-token activation color      per-token projection onto region direction
    activation histogram            projection + distance-to-exemplar histograms
    logits (pos/neg)                logit lens of exemplar & mean (+ J-lens)
    frequency / sparsity            member count, density, coherence
    correlated features             nearest regions by exemplar cosine

Outputs a batched raw-JSON dataset plus self-contained local HTML pages.
See DECISIONS.md for methodology notes.
"""

from epdashboard.config import EPVisConfig

__version__ = "0.1.0"
__all__ = ["EPVisConfig", "__version__"]
