from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Particle:
    """
    Minimal four-momentum carrier with PUPPI weight and truth tags.
    """

    px: float
    py: float
    pz: float
    e: float
    charge: float
    is_lv: bool
    weight: float = 1.0

    @property
    def pt(self) -> float:
        return float(np.sqrt(self.px**2 + self.py**2))

    @property
    def eta(self) -> float:
        p = np.sqrt(self.px**2 + self.py**2 + self.pz**2)

        if p == 0 or p == abs(self.pz):
            return float(np.sign(self.pz) * 1e9)

        return float(0.5 * np.log((p + self.pz) / (p - self.pz)))

    @property
    def phi(self) -> float:
        return float(np.arctan2(self.py, self.px))

    @property
    def mass(self) -> float:
        m2 = self.e**2 - self.px**2 - self.py**2 - self.pz**2
        return float(np.sqrt(max(m2, 0.0)))

    def rescaled(self) -> "Particle":
        """
        Return a new particle with four-momentum scaled by the PUPPI weight.
        """
        w = self.weight

        return Particle(
            px=self.px * w,
            py=self.py * w,
            pz=self.pz * w,
            e=self.e * w,
            charge=self.charge,
            is_lv=self.is_lv,
            weight=self.weight,
        )


def unpack_particles(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    e: np.ndarray,
    charge: np.ndarray,
    is_lv: np.ndarray,
    n: int,
) -> list[Particle]:
    """
    Convert zero-padded full-event arrays into a list of Particle objects.
    """
    particles: list[Particle] = []

    for i in range(int(n)):
        particles.append(
            Particle(
                px=float(px[i]),
                py=float(py[i]),
                pz=float(pz[i]),
                e=float(e[i]),
                charge=float(charge[i]),
                is_lv=bool(is_lv[i] > 0.5),
            )
        )

    return particles