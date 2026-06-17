#!/usr/bin/env python3
"""
siRNADiscovery prediction interface for the siRNA-split model.

Predicts siRNA efficacy for siRNA-mRNA sequence pairs using a trained
HinSAGE model.

The model requires both siRNA and mRNA sequences because the GNN has three
node types: siRNA, mRNA, and interaction.  Features are split into two
categories:

  * Sequence-derived (computed on the fly):
      siRNA:  one-hot (84), self-fold (6), AGO2 (1), GC% (1),
              k-mers 1-5 (1364), rules scores (57)  →  1513 total
      mRNA:   one-hot (39024), self-fold (100), AGO2 (1), GC% (1)
              →  39126 total
      Inter.: thermodynamics (21), co-fold (50), pos-encoding (126)
              →  197 total

  * External-tool-dependent (loaded from precomputed files or zero-filled):
      - siRNA self-fold (RNAfold→SVD, 6-d)
      - mRNA  self-fold (RNAfold→SVD, 100-d)
      - siRNA-mRNA co-fold (RNAcofold→SVD, 50-d)
      - siRNA-AGO2 / mRNA-AGO2 interaction prob. (RPISeq, 1-d each)

Usage examples:

  # Single pair
  python predict.py --sirna-seq UUGCUAGAGAGUUUGGUGUU \\
                    --mrna-seq CAGCAGUGGCAGUUCAGAUGCG... \\
                    --model-weights saved_models/fold0_weights.h5

  # Batch from CSV (columns: siRNA_seq, mRNA_seq)
  python predict.py --csv input.csv --output preds.csv \\
                    --model-weights saved_models/fold0_weights.h5

  # Interactive one-shot (stdin CSV without header)
  echo "UUGCU...,CAGCA..." | python predict.py \\
      --model-weights saved_models/fold0_weights.h5

  # Ensemble all 10 folds via glob
  python predict.py --csv input.csv --output ens_preds.csv \\
      --model-weights "saved_models/fold*.h5"

  # Ensemble specific folds
  python predict.py --csv input.csv \\
      --model-weights "fold0.h5,fold1.h5,fold2.h5"
"""

import argparse, json, os, sys, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import mean_squared_error, roc_auc_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import stellargraph as StellarGraph
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE

import utils

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
_DEFAULT_PARAMS = {
    "dmodel": 6,
    "sirna_length": 21,
    "max_mrna_len": 9756,
    "batch_size": 64,
    "hinsage_layer_sizes": [64, 32],
    "hop_samples": [12, 6],
}
_SELF_SIRNA_COLS = 6
_SELF_MRNA_COLS = 100
_CON_COLS = 50

# Set at runtime by reading the weight file
_INTERACTION_COLS = None


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _make_id(seq_label, idx):
    """Short unique id for a sequence (used as graph node key)."""
    return f"s{idx}_{seq_label[:8]}"


def _reverse_complement(seq: str) -> str:
    """Reverse-complement DNA (A↔T, C↔G)."""
    trans = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq[::-1].translate(trans)


def compute_sirna_features(seq: str) -> np.ndarray:
    """Compute all sequence-derived siRNA features from raw sequence (DNA)."""
    seq_t = seq.replace("U", "T")
    oh = utils.obtain_one_hot_feature_for_one_sequence_1(seq_t, _DEFAULT_PARAMS["sirna_length"])
    gc = utils.countGC(seq_t)
    k1 = utils.single_freq(seq_t)
    k2 = utils.double_freq(seq_t)
    k3 = utils.triple_freq(seq_t)
    k4 = utils.quadruple_freq(seq_t)
    k5 = utils.quintuple_freq(seq_t)
    rs = utils.rules_scores(seq_t)
    k_all = {**k1, **k2, **k3, **k4, **k5}
    vals = np.concatenate([
        np.asarray(oh, dtype=np.float32),
        np.zeros(_SELF_SIRNA_COLS, dtype=np.float32),  # self-fold placeholder
        np.zeros(1, dtype=np.float32),                  # AGO2 placeholder
        np.asarray([gc], dtype=np.float32),
        np.asarray(list(k_all.values()), dtype=np.float32),
        np.asarray(rs, dtype=np.float32),
    ])
    return vals


def compute_mrna_features(seq: str) -> np.ndarray:
    """Compute all sequence-derived mRNA features from raw sequence."""
    oh = utils.obtain_one_hot_feature_for_one_sequence_1(seq, _DEFAULT_PARAMS["max_mrna_len"])
    gc = utils.countGC(seq)
    vals = np.concatenate([
        np.asarray(oh, dtype=np.float32),
        np.zeros(_SELF_MRNA_COLS, dtype=np.float32),  # self-fold placeholder
        np.zeros(1, dtype=np.float32),                 # AGO2 placeholder
        np.asarray([gc], dtype=np.float32),
    ])
    return vals


def compute_interaction_features(sirna_seq: str, mrna_seq: str,
                                  match_pos: int) -> np.ndarray:
    """Compute interaction-node features for a (siRNA, mRNA) pair."""
    global _INTERACTION_COLS
    seq_u = sirna_seq.replace("T", "U")
    thermo = utils.cal_thermo_feature(seq_u)
    pe = utils.get_pos_embedding_sequence(
        match_pos, _DEFAULT_PARAMS["sirna_length"], _DEFAULT_PARAMS["dmodel"])
    con = np.zeros(_CON_COLS, dtype=np.float32)
    raw = np.concatenate([
        np.asarray(thermo, dtype=np.float32), con,
        np.asarray(pe, dtype=np.float32),
    ])
    # Trim or pad to match the trained model's expected dimension
    if _INTERACTION_COLS is not None and raw.shape[0] != _INTERACTION_COLS:
        if len(raw) > _INTERACTION_COLS:
            raw = raw[:_INTERACTION_COLS]
        else:
            raw = np.pad(raw, (0, _INTERACTION_COLS - len(raw)))
    return raw


def find_match_position(sirna_seq: str, mrna_seq: str) -> int:
    """Find where the reverse-complement of siRNA aligns in the mRNA."""
    rc = _reverse_complement(sirna_seq.replace("U", "T"))
    return mrna_seq.index(rc)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_prediction_graph(records: list) -> StellarGraph.StellarGraph:
    """Build a StellarGraph from a list of (sirna_id, sirna_seq, mrna_id, mrna_seq) tuples.

    Each record becomes an interaction node; siRNA and mRNA nodes are
    deduplicated across records.
    """
    # Deduplicate siRNA / mRNA by sequence (use first occurrence's id)
    sirna_map = {}  # seq -> (id, features)
    mrna_map = {}

    interaction_rows = []
    sirna_rows = []
    mrna_rows = []
    edges = []

    for idx, (s_label, s_seq, m_label, m_seq) in enumerate(records):
        s_seq_t = s_seq.replace("U", "T")
        m_seq_t = m_seq.replace("U", "T")

        # siRNA node
        if s_seq_t not in sirna_map:
            sid = _make_id(s_label, idx)
            sirna_map[s_seq_t] = (sid, compute_sirna_features(s_seq_t))
        else:
            sid = sirna_map[s_seq_t][0]

        # mRNA node
        if m_seq_t not in mrna_map:
            mid = _make_id(m_label, idx)
            mrna_map[m_seq_t] = (mid, compute_mrna_features(m_seq_t))
        else:
            mid = mrna_map[m_seq_t][0]

        # Interaction node
        iid = f"i{idx}_{sid}_{mid}"
        try:
            match_pos = find_match_position(s_seq_t, m_seq_t)
        except ValueError:
            print(f"Warning: siRNA {s_label} does not reverse-complement-match "
                  f"mRNA {m_label}; using position 0.", file=sys.stderr)
            match_pos = 0
        interaction_feat = compute_interaction_features(s_seq_t, m_seq_t, match_pos)
        interaction_rows.append((iid, interaction_feat))

        # Edges interaction → siRNA, interaction → mRNA
        edges.append((iid, sid))
        edges.append((iid, mid))

    # Build DataFrames — index must be the node ID (v[0]), not the sequence (k)
    sirna_df = pd.DataFrame({v[0]: v[1] for v in sirna_map.values()}).T
    sirna_df.index.name = "siRNA"

    mrna_df = pd.DataFrame({v[0]: v[1] for v in mrna_map.values()}).T
    mrna_df.index.name = "mRNA"

    interaction_df = pd.DataFrame(
        {iid: feat for iid, feat in interaction_rows}).T
    interaction_df.index.name = "interaction"

    edge_df = pd.DataFrame(edges, columns=["source", "target"])

    return StellarGraph.StellarGraph(
        {"siRNA": sirna_df, "mRNA": mrna_df, "interaction": interaction_df},
        edges=edge_df, source_column="source", target_column="target"
    )


# ---------------------------------------------------------------------------
# Weight introspection (detect expected feature dimensions)
# ---------------------------------------------------------------------------

def detect_feature_dims(weight_path: str):
    """Read the trained weight file to determine expected feature dimensions.

    Scans all aggregator w_self weights and picks the interaction dimension
    (the one that is not a known siRNA/mRNA/hidden dimension).
    """
    global _INTERACTION_COLS
    known_other = {1513, 39126, 64, 32, 16}
    candidates = []
    try:
        import h5py
        with h5py.File(weight_path, "r") as f:
            def _visitor(path, obj):
                if isinstance(obj, h5py.Dataset) and path.endswith("/w_self:0"):
                    candidates.append(obj.shape[0])
            f.visititems(_visitor)
        print(f"All aggregator w_self dims found: {sorted(set(candidates))}",
              file=sys.stderr)
        for d in sorted(set(candidates)):
            if d not in known_other:
                _INTERACTION_COLS = d
                break
        if _INTERACTION_COLS is None and candidates:
            _INTERACTION_COLS = candidates[0]
        if _INTERACTION_COLS is not None:
            print(f"Detected interaction feature dim: {_INTERACTION_COLS}",
                  file=sys.stderr)
    except Exception as e:
        print(f"Could not probe weight file ({e}); using defaults.",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Model building (must match training architecture)
# ---------------------------------------------------------------------------

def build_inference_model(generator: HinSAGENodeGenerator,
                          params: dict) -> Model:
    """Build the same Keras model architecture used during training."""
    hinsage = HinSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        generator=generator,
        bias=True,
        dropout=0.0,  # no dropout for inference
    )
    x_inp, x_out = hinsage.in_out_tensors()
    prediction = layers.Dense(units=1)(x_out)
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer=optimizers.Adam(learning_rate=params["lr"]),
                  loss=params["loss"])
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="siRNADiscovery prediction interface (siRNA split)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Single pair
              %(prog)s --sirna-seq UUGCUAGAGAGUUUGGUGUU \\
                        --mrna-seq CAGCAGUGGCAGUUC... \\
                        --model-weights fold0_weights.h5

              # Batch from CSV
              %(prog)s --csv input_pairs.csv --output preds.csv \\
                        --model-weights fold0_weights.h5

              # Interactive: one pair per line, CSV format (siRNA_seq,mRNA_seq)
              echo "UUGCU...,CAGCA..." | %(prog)s --model-weights fold0_weights.h5
        """),
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--sirna-seq", type=str, default=None,
                          help="siRNA sequence (DNA, 21 nt)")
    io_group.add_argument("--mrna-seq", type=str, default=None,
                          help="mRNA sequence (DNA)")
    io_group.add_argument("--csv", type=str, default=None,
                          help="Path to CSV with columns: siRNA_seq,mRNA_seq")
    io_group.add_argument("--output", type=str, default=None,
                          help="Path for output CSV (default: stdout)")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model-weights", type=str, required=True,
                             help="Path(s) to .h5 weights, comma-separated or glob, e.g. "
                                  "'saved_models/fold*.h5' or 'fold0.h5,fold1.h5'")
    model_group.add_argument("--params", type=str, default=None,
                             help="Path to siRNA_param.json (default: siRNA_param.json in same dir)")

    return parser.parse_args(argv)


def main():
    args = parse_args()

    # ---- Load params ----
    if args.params is not None:
        params_path = args.params
    else:
        params_path = Path(__file__).parent / "siRNA_param.json"
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = {**_DEFAULT_PARAMS, **json.load(f)}
    else:
        print(f"Parameter file not found at {params_path}, using defaults.")
        params = _DEFAULT_PARAMS.copy()
    params.setdefault("lr", 0.001)
    params.setdefault("loss", "mse")

    # ---- Collect input sequences ----
    records = []  # list of (sirna_label, sirna_seq, mrna_label, mrna_seq)

    if args.sirna_seq and args.mrna_seq:
        records.append(("input_siRNA", args.sirna_seq,
                        "input_mRNA", args.mrna_seq))

    if args.csv:
        df_in = pd.read_csv(args.csv)
        for _, row in df_in.iterrows():
            records.append((
                row.get("siRNA", f"siRNA_{_}"),
                row.get("siRNA_seq", row.get("sirna_seq")),
                row.get("mRNA", f"mRNA_{_}"),
                row.get("mRNA_seq", row.get("mrna_seq")),
            ))

    if not records and not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                records.append((f"siRNA_{len(records)}", parts[0].strip(),
                                f"mRNA_{len(records)}", parts[1].strip()))

    if not records:
        print("Error: no input sequences provided.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(records)} siRNA-mRNA pair(s)...", file=sys.stderr)

    # ---- Resolve weight files (comma-separated or glob) ----
    if "," in args.model_weights and not any(
        c in args.model_weights for c in "*?["
    ):
        weight_files = [p.strip() for p in args.model_weights.split(",")]
    else:
        import glob as _glob
        weight_files = sorted(_glob.glob(args.model_weights))
        if not weight_files:
            weight_files = [args.model_weights]
    print(f"Using {len(weight_files)} model(s): "
          f"{[os.path.basename(w) for w in weight_files]}", file=sys.stderr)

    # ---- Run prediction for each weight file ----
    all_preds = []
    for wf in weight_files:
        detect_feature_dims(wf)
        print(f"  Building graph for {os.path.basename(wf)}...", file=sys.stderr)
        g = build_prediction_graph(records)
        gen = HinSAGENodeGenerator(
            g, params["batch_size"], params["hop_samples"],
            head_node_type="interaction",
        )
        iids = list(g.nodes(node_type="interaction"))
        dummy = pd.DataFrame(np.zeros((len(iids), 1)), index=iids)
        model = build_inference_model(gen, params)
        model.load_weights(wf)
        preds = np.squeeze(model.predict(gen.flow(dummy.index, dummy), verbose=0))
        all_preds.append(preds)

    # ---- Ensemble (average across folds) ----
    if len(all_preds) > 1:
        preds = np.mean(all_preds, axis=0)
        print(f"Ensembled {len(all_preds)} models via averaging", file=sys.stderr)
    else:
        preds = all_preds[0]

    # ---- Output ----
    results = pd.DataFrame({
        "siRNA": [r[0] for r in records],
        "mRNA": [r[2] for r in records],
        "predicted_efficacy": preds,
    })
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Predictions written to {args.output}", file=sys.stderr)
    else:
        results.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
