"""
MadGraph stage for the PileFlow generator.

This module wraps MadGraph5_aMC@NLO execution.

It supports:
    1. Interactive MG5 execution.
    2. Automatic MG5 execution using a generated script.
    3. LHE file discovery inside MG5 output folders.
    4. Feynman diagram collection from MG5 process folders.

Important
---------
This stage should remain optional during migration.

For reproducibility debugging, prefer running from a frozen LHE file first.
Only use automatic MG5 once the fixed-LHE path reproduces the old reference.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import Any


class MadGraphRunner:
    """
    Helper class for managing MadGraph execution and outputs.

    Parameters
    ----------
    mg5_path:
        Path to the ``mg5_aMC`` executable.

    work_dir:
        Directory where MG5 scripts and process folders are written.
    """

    def __init__(self, mg5_path: str, work_dir: str):
        self.mg5_path = mg5_path
        self.work_dir = work_dir

    def _check_mg5_exists(self) -> None:
        """
        Check that the configured MG5 executable exists.

        Raises
        ------
        FileNotFoundError
            If ``self.mg5_path`` does not point to a file.
        """
        if not os.path.isfile(self.mg5_path):
            raise FileNotFoundError(
                f"Could not find mg5_aMC at:\n"
                f"  {self.mg5_path}\n"
                "Please fix the path or export MG5_PATH."
            )

    def _build_mg5_script(
        self,
        process_name: str,
        process_command: str,
        mg5_nevents: int,
        mg5_ptj: float | None = None,
        extra_run_card_edits: dict[str, Any] | None = None,
    ) -> str:
        """
        Build a temporary MadGraph script for automatic event generation.

        Parameters
        ----------
        process_name:
            Output directory name used by MG5, for example ``pp_jj``.

        process_command:
            MG5 process command, for example ``generate p p > j j``.

        mg5_nevents:
            Number of events written into the MG5 run card.

        mg5_ptj:
            Optional minimum jet transverse-momentum cut written as ``set ptj``.

        extra_run_card_edits:
            Additional run-card edits written as ``set key value`` lines.

        Returns
        -------
        str
            Full MG5 script content.
        """
        if extra_run_card_edits is None:
            extra_run_card_edits = {}

        lines: list[str] = [
            process_command,
            f"output {process_name}",
            f"launch {process_name}",
            f"set nevents {int(mg5_nevents)}",
        ]

        if mg5_ptj is not None:
            lines.append(f"set ptj {float(mg5_ptj)}")

        for key, value in extra_run_card_edits.items():
            lines.append(f"set {key} {value}")

        lines.append("done")

        return "\n".join(lines) + "\n"

    def run_interactive(self) -> None:
        """
        Launch MadGraph in fully interactive mode.

        This opens MG5 and leaves the process generation, card editing, and
        launch choices to the user inside the MG5 prompt.
        """
        self._check_mg5_exists()

        print("\nLaunching MadGraph interactively...")
        print(
            "Generate your process, create the output folder, launch, "
            "edit cards if needed, then quit.\n"
        )

        subprocess.run([self.mg5_path], check=False)

    def run_automatic(
        self,
        process_name: str,
        process_command: str,
        mg5_nevents: int,
        mg5_ptj: float | None = None,
        extra_run_card_edits: dict[str, Any] | None = None,
    ) -> None:
        """
        Run MadGraph automatically using a generated MG5 script.

        Parameters
        ----------
        process_name:
            MG5 output folder name.

        process_command:
            MG5 process command.

        mg5_nevents:
            Number of MG5 events to write into the LHE file.

        mg5_ptj:
            Optional ``ptj`` run-card cut.

        extra_run_card_edits:
            Extra run-card settings.
        """
        self._check_mg5_exists()
        os.makedirs(self.work_dir, exist_ok=True)

        script_content = self._build_mg5_script(
            process_name=process_name,
            process_command=process_command,
            mg5_nevents=mg5_nevents,
            mg5_ptj=mg5_ptj,
            extra_run_card_edits=extra_run_card_edits,
        )

        script_path = os.path.join(self.work_dir, f"mg5_auto_{process_name}.txt")

        with open(script_path, "w") as f:
            f.write(script_content)

        print("\nLaunching MadGraph automatically...")
        print(f"MG5 script: {script_path}")

        subprocess.run([self.mg5_path, script_path], check=True)

    def find_lhe_file(self, process_name: str) -> str | None:
        """
        Search for an LHE file inside common MadGraph output locations.

        Parameters
        ----------
        process_name:
            Name of the MG5 process/output directory.

        Returns
        -------
        str | None
            Path to the newest matching LHE file, or ``None`` if no file is found.
        """
        patterns = [
            os.path.join(
                self.work_dir,
                process_name,
                "Events",
                "run_01",
                "unweighted_events.lhe.gz",
            ),
            os.path.join(
                self.work_dir,
                process_name,
                "Events",
                "run_01",
                "unweighted_events.lhe",
            ),
            os.path.join(
                self.work_dir,
                "**",
                process_name,
                "**",
                "unweighted_events.lhe.gz",
            ),
            os.path.join(
                self.work_dir,
                "**",
                process_name,
                "**",
                "unweighted_events.lhe",
            ),
        ]

        hits: list[str] = []

        for pattern in patterns:
            hits.extend(glob.glob(pattern, recursive=True))

        hits = sorted(set(hits), key=os.path.getmtime, reverse=True)

        return hits[0] if hits else None

    def collect_feynman_diagrams(self, process_name: str, run_dir: str) -> None:
        """
        Copy Feynman diagrams produced by MG5 into the workflow output folder.

        Parameters
        ----------
        process_name:
            MG5 process/output directory name.

        run_dir:
            Timestamped run directory where copied diagrams should be stored.
        """
        proc_candidates = [
            os.path.join(self.work_dir, process_name),
            os.path.join(self.work_dir, os.path.basename(process_name)),
        ]

        proc_dir = None

        for candidate in proc_candidates:
            if os.path.isdir(candidate):
                proc_dir = candidate
                break

        if proc_dir is None:
            print("No MadGraph process directory found for diagram collection.")
            return

        out_dir = os.path.join(run_dir, "feynman_diagrams")
        os.makedirs(out_dir, exist_ok=True)

        patterns = [
            os.path.join(proc_dir, "SubProcesses", "**", "*.pdf"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.png"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.jpg"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.jpeg"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.eps"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.ps"),
            os.path.join(proc_dir, "HTML", "**", "*.pdf"),
            os.path.join(proc_dir, "HTML", "**", "*.png"),
            os.path.join(proc_dir, "HTML", "**", "*.jpg"),
            os.path.join(proc_dir, "HTML", "**", "*.jpeg"),
            os.path.join(proc_dir, "HTML", "**", "*.eps"),
            os.path.join(proc_dir, "HTML", "**", "*.ps"),
        ]

        found: list[str] = []

        for pattern in patterns:
            found.extend(glob.glob(pattern, recursive=True))

        found = sorted(set(found))

        if not found:
            print("No Feynman diagrams found.")
            return

        copied = 0

        for src in found:
            name = os.path.basename(src)
            dst = os.path.join(out_dir, name)

            if os.path.exists(dst):
                base, ext = os.path.splitext(name)
                k = 1

                while True:
                    candidate = os.path.join(out_dir, f"{base}_{k}{ext}")

                    if not os.path.exists(candidate):
                        dst = candidate
                        break

                    k += 1

            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as exc:
                print(f"Failed to copy diagram {src}: {exc}")

        print(f"Copied {copied} diagram files into: {out_dir}")