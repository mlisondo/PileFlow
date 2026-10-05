"""Runs the full pipeline on a tiny synthetic dataset (CPU, ~1 min)."""
import json

import numpy as np


def test_full_pipeline(synthetic_npz, tmp_path):
    import compare_methods
    import evaluate_checkpoint
    import select_threshold
    from pileflow import run
    from pumml import evaluate as pumml_eval
    from pumml import run_pumml

    data = str(synthetic_npz)
    rundir = tmp_path / "pf"
    run.main(["--data", data, "--outdir", str(rundir), "--arch", "spatial_small", "--device", "cpu",
              "--epochs", "3", "--batch-size", "64", "--ode-solver", "euler", "--ode-steps", "4",
              "--eval-splits", "eval", "--no-plots", "--morph-lambda", "0.1"])
    ckpt = rundir / "checkpoints" / "pileflow_best.pt"
    assert ckpt.exists() and (rundir / "config.json").exists()

    evaluate_checkpoint.main(["--data", data, "--checkpoint", str(ckpt), "--outdir", str(rundir),
                              "--split", "val", "--device", "cpu", "--eval-samples", "3",
                              "--ode-solver", "midpoint", "--ode-steps", "3", "--no-plots"])
    evaluate_checkpoint.main(["--data", data, "--checkpoint", str(ckpt), "--outdir", str(rundir),
                              "--split", "eval", "--device", "cpu", "--eval-samples", "3",
                              "--ode-solver", "midpoint", "--ode-steps", "3", "--no-plots"])
    m = json.loads((rundir / "metrics_eval.json").read_text())
    assert 0 <= m["neutral_pt_coverage_90"] <= 1

    # reproducibility of sampling
    a = np.load(rundir / "data" / "generated_eval.npz")["neutral_lv_pred"]
    evaluate_checkpoint.main(["--data", data, "--checkpoint", str(ckpt), "--outdir", str(tmp_path / "again"),
                              "--split", "eval", "--device", "cpu", "--eval-samples", "3", "--batch-size", "7",
                              "--ode-solver", "midpoint", "--ode-steps", "3", "--no-plots"])
    b = np.load(tmp_path / "again" / "data" / "generated_eval.npz")["neutral_lv_pred"]
    assert np.allclose(a, b, atol=1e-5)

    thr = rundir / "threshold.json"
    select_threshold.main(["--data", data, "--generated", str(rundir / "data" / "generated_val.npz"),
                           "--output", str(thr), "--grid", "0", "0.1", "0.5"])
    assert json.loads(thr.read_text())["selected_on_split"] == "val"

    pm = tmp_path / "pumml.pt"
    run_pumml.main(["--data", data, "--epochs", "2", "--device", "cpu", "--output", str(pm)])
    pm2 = tmp_path / "pumml_tuned.pt"
    run_pumml.main(["--data", data, "--epochs", "2", "--device", "cpu", "--select-on-val", "--output", str(pm2)])

    pumml_eval.main(["--data", data, "--model", str(pm), "--output-dir", str(tmp_path / "repro"),
                     "--device", "cpu", "--bootstrap-samples", "20"])
    out = tmp_path / "cmp"
    compare_methods.main(["--data", data, "--pumml-model", str(pm), "--pumml-tuned-model", str(pm2),
                          "--pileflow", str(rundir / "data" / "generated_eval.npz"),
                          "--threshold-json", str(thr), "--add-posterior-mean", "--device", "cpu",
                          "--bootstrap-samples", "30", "--spread-samples", "2", "--outdir", str(out)])
    for variant in ("nocut", "cut_100MeV"):
        summary = json.loads((out / variant / "tables" / "summary.json").read_text())
        assert "diff_vs_pumml" in summary["correlation"]["jet_mass"]
        assert (out / variant / "plots" / "fig4_observable_distributions.png").exists()
        assert (out / variant / "tables" / "correlation_table.tex").exists()
    rows = json.loads((out / "tables" / "cut_vs_nocut.json").read_text())["rows"]
    assert {r["method"] for r in rows} >= {"pumml", "pileflow"}
    assert (out / "plots" / "cut_vs_nocut_correlations.png").exists()
    assert (out / "plots" / "cut_vs_nocut_n95.png").exists()
    assert (tmp_path / "repro" / "tables" / "cut_vs_nocut.csv").exists()

    import threshold_scan
    threshold_scan.main(["--data", data, "--pumml-model", str(pm), "--pileflow",
                         str(rundir / "data" / "generated_eval.npz"), "--device", "cpu", "--grid", "0", "0.05", "0.2",
                         "--bootstrap-samples", "20", "--outdir", str(tmp_path / "scan")])
    assert (tmp_path / "scan" / "threshold_scan_summary.json").exists()
    compare_methods.main(["--data", data, "--pumml-model", str(pm), "--pileflow",
                          str(rundir / "data" / "generated_eval.npz"), "--cut-mode", "off",
                          "--device", "cpu", "--bootstrap-samples", "10", "--spread-samples", "0",
                          "--outdir", str(tmp_path / "cmp_off")])
    assert (tmp_path / "cmp_off" / "nocut").exists() and not (tmp_path / "cmp_off" / "cut_100MeV").exists()
    compare_methods.main(["--data", data, "--pumml-model", str(pm), "--pileflow",
                          str(rundir / "data" / "generated_eval.npz"), "--cut-mode", "on", "--cell-threshold",
                          "0.085", "--threshold-truth", "--device", "cpu", "--bootstrap-samples", "10",
                          "--spread-samples", "0", "--outdir", str(tmp_path / "cmp_on")])
    assert (tmp_path / "cmp_on" / "cut_85MeV").exists() and not (tmp_path / "cmp_on" / "nocut").exists()

    import multiseed_benchmark
    multiseed_benchmark.main(["--data", data, "--pumml-models", str(pm), str(pm2),
                              "--pileflow", str(rundir / "data" / "generated_eval.npz"),
                              str(rundir / "data" / "generated_eval.npz"), "--device", "cpu",
                              "--bootstrap-samples", "10", "--n95-scan", "--outdir", str(tmp_path / "ms")])
    for v in ("nocut", "cut_100MeV"):
        assert (tmp_path / "ms" / v / "tables" / "table1_correlation.tex").exists()
        assert (tmp_path / "ms" / v / "reference_seed" / "plots" / "fig4_observable_distributions.png").exists()
    assert (tmp_path / "ms" / "appendix_n95_threshold_all_seeds.png").exists()

    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "plot_average_channels.py"), "--data", data,
                    "--outdir", str(tmp_path / "avg")], check=True, cwd=root)
    subprocess.run([sys.executable, str(root / "plot_qualitative_jets.py"), "--data", data,
                    "--pumml-model", str(pm), "--pileflow", str(rundir / "data" / "generated_eval.npz"),
                    "--outdir", str(tmp_path / "qual"), "--num-jets", "2", "--device", "cpu",
                    "--dpi", "60", "--selection", "random", "--cell-threshold", "0.1"], check=True, cwd=root)
