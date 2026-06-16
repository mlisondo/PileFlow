"""
Jet and parton matching utilities.

This module contains matching rules used by reconstruction and image-building
stages. It should stay independent of FastJet event loops and output schemas.
"""

from __future__ import annotations

from typing import Sequence

from pileflow_generator.physics.kinematics import delta_r


# Higher score wins when more than one parton is inside the matching cone.
#
# This preserves the old priority rule:
#     b > c > t > light quarks > gluon
PARTON_MATCH_PRIORITY: dict[int, int] = {
    5: 4,   # b
    4: 3,   # c
    6: 2,   # t
    1: 1,   # d
    2: 1,   # u
    3: 1,   # s
    21: 0,  # gluon
}


def match_flavour(
    jet_eta: float,
    jet_phi: float,
    partons: Sequence[tuple[int, float, float, float]],
    R: float,
) -> int:
    """
    Match a jet to a hard parton using a two-cone priority rule.

    Parameters
    ----------
    jet_eta, jet_phi:
        Jet axis coordinates.

    partons:
        Sequence of ``(abs_pid, eta, phi, pt)`` tuples.

    R:
        Jet radius parameter.

    Returns
    -------
    int
        Absolute PDG ID of the matched parton.
        Returns 0 if no parton is matched.

    Notes
    -----
    1. First search inside a strict cone ``max(0.2, 0.4 * R)``.
    2. If nothing is found, search inside the full jet radius ``R``.
    3. If several partons are inside the cone, choose by priority:
       b > c > t > light quarks > gluon.
    4. If priority ties, choose the smaller Delta R.
    """

    def best_in_cone(dr_max: float) -> int:
        best_pid = 0
        best_priority = -1
        best_dr = float("inf")

        for pid, p_eta, p_phi, _pt in partons:
            abs_pid = abs(int(pid))
            dr = delta_r(jet_eta, jet_phi, p_eta, p_phi)
            priority = PARTON_MATCH_PRIORITY.get(abs_pid, 0)

            if dr < dr_max and (
                priority > best_priority
                or (priority == best_priority and dr < best_dr)
            ):
                best_pid = abs_pid
                best_priority = priority
                best_dr = dr

        return int(best_pid)

    strict_cone = max(0.2, 0.4 * R)
    pid = best_in_cone(strict_cone)

    if pid == 0:
        pid = best_in_cone(R)

    return int(pid)