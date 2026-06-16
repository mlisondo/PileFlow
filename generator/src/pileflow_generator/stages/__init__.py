"""
Pipeline stages for the PileFlow generator.
"""

from pileflow_generator.stages.madgraph import MadGraphRunner
from pileflow_generator.stages.pythia import PythiaRunner

from pileflow_generator.stages.temporary_baseline_puppi import (
    Particle,
    unpack_particles,
    characterise_pileup,
    run_puppi,
    run_puppi_on_dataset,
)

from pileflow_generator.stages.pileup import (
    TaggedParticle,
    PileupOverlay,
    tagged_from_snapshot,
)

from pileflow_generator.stages.images import (
    JetImageBuilder,
    produce_images,
)

from pileflow_generator.stages.clustering import FastJetRunner

from pileflow_generator.stages.features import (
    build_feature_row,
    compute_btag,
    compute_ctag,
    compute_fractions,
    jet_quality_id,
    quark_gluon_likelihood,
)

__all__ = [
    "MadGraphRunner",
    "PythiaRunner",

    "Particle",
    "unpack_particles",
    "characterise_pileup",
    "run_puppi",
    "run_puppi_on_dataset",

    "TaggedParticle",
    "PileupOverlay",
    "tagged_from_snapshot",

    "JetImageBuilder",
    "produce_images",

    "FastJetRunner",

    "build_feature_row",
    "compute_btag",
    "compute_ctag",
    "compute_fractions",
    "jet_quality_id",
    "quark_gluon_likelihood",
]