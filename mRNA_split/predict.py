#!/usr/bin/env python3
"""
siRNADiscovery prediction interface for the mRNA-split model.

Usage: see siRNA_split/predict.py for examples.  The only differences
are the parameter values and feature dimensionalities:
  - self-siRNA fold: 15 components
  - self-mRNA  fold: 500 components
  - dmodel: 3
  - hop_samples: [4, 2]
  - dropout: 0.3
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

_DEFAULT_PARAMS = {
    "dmodel": 3,
    "sirna_length": 21,
    "max_mrna_len": 9756,
    "batch_size": 64,
    "hinsage_layer_sizes": [64, 32],
    "hop_samples": [4, 2],
}
_PREPROCESS_DIR = Path(__file__).parent / "mRNA_split_preprocess"
_AGO2_DIR = Path(__file__).parent / "RNA_AGO2"
_SELF_SIRNA_COLS = 15
_SELF_MRNA_COLS = 500
_CON_COLS = 50


def _make_id(seq_label, idx):
    return f"s{idx}_{seq_label[:8]}"


def _reverse_complement(seq: str) -> str:
    trans = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq[::-1].translate(trans)


def compute_sirna_features(seq: str) -> np.ndarray:
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
        np.zeros(_SELF_SIRNA_COLS, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.asarray([gc], dtype=np.float32),
        np.asarray(list(k_all.values()), dtype=np.float32),
        np.asarray(rs, dtype=np.float32),
    ])
    return vals


def compute_mrna_features(seq: str) -> np.ndarray:
    oh = utils.obtain_one_hot_feature_for_one_sequence_1(seq, _DEFAULT_PARAMS["max_mrna_len"])
    gc = utils.countGC(seq)
    vals = np.concatenate([
        np.asarray(oh, dtype=np.float32),
        np.zeros(_SELF_MRNA_COLS, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.asarray([gc], dtype=np.float32),
    ])
    return vals


def compute_interaction_features(sirna_seq: str, mrna_seq: str,
                                  match_pos: int) -> np.ndarray:
    seq_u = sirna_seq.replace("T", "U")
    thermo = utils.cal_thermo_feature(seq_u)
    pe = utils.get_pos_embedding_sequence(
        match_pos, _DEFAULT_PARAMS["sirna_length"], _DEFAULT_PARAMS["dmodel"])
    vals = np.concatenate([
        np.asarray(thermo, dtype=np.float32),
        np.zeros(_CON_COLS, dtype=np.float32),
        np.asarray(pe, dtype=np.float32),
    ])
    return vals


def find_match_position(sirna_seq: str, mrna_seq: str) -> int:
    rc = _reverse_complement(sirna_seq.replace("U", "T"))
    return mrna_seq.index(rc)


def build_prediction_graph(records: list) -> StellarGraph.StellarGraph:
    """Build a StellarGraph from (sirna_id, sirna_seq, mrna_id, mrna_seq) tuples."""
    sirna_map = {}
    mrna_map = {}
    interaction_rows = []
    edges = []

    for idx, (s_label, s_seq, m_label, m_seq) in enumerate(records):
        s_seq_t = s_seq.replace("U", "T")
        m_seq_t = m_seq.replace("U", "T")

        if s_seq_t not in sirna_map:
            sid = _make_id(s_label, idx)
            sirna_map[s_seq_t] = (sid, compute_sirna_features(s_seq_t))
        else:
            sid = sirna_map[s_seq_t][0]

        if m_seq_t not in mrna_map:
            mid = _make_id(m_label, idx)
            mrna_map[m_seq_t] = (mid, compute_mrna_features(m_seq_t))
        else:
            mid = mrna_map[m_seq_t][0]

        iid = f"i{idx}_{sid}_{mid}"
        try:
            match_pos = find_match_position(s_seq_t, m_seq_t)
        except ValueError:
            print(f"Warning: siRNA {s_label} does not reverse-complement-match "
                  f"mRNA {m_label}; using position 0.", file=sys.stderr)
            match_pos = 0
        interaction_feat = compute_interaction_features(s_seq_t, m_seq_t, match_pos)
        interaction_rows.append((iid, interaction_feat))
        edges.append((iid, sid))
        edges.append((iid, mid))

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


def build_inference_model(generator: HinSAGENodeGenerator,
                          params: dict) -> Model:
    hinsage = HinSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        generator=generator,
        bias=True,
        dropout=0.0,
    )
    x_inp, x_out = hinsage.in_out_tensors()
    prediction = layers.Dense(units=1)(x_out)
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer=optimizers.Adam(learning_rate=params.get("lr", 0.001)),
                  loss=params.get("loss", "mse"))
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="siRNADiscovery prediction interface (mRNA split)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python predict.py --sirna-seq UUGCUAGAGAGUUUGGUGUU \\
                  --mrna-seq CAGCAG... --model-weights fold0_weights.h5
              python predict.py --csv input.csv --output preds.csv \\
                  --model-weights fold0_weights.h5
        """),
    )
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--sirna-seq", type=str, default=None)
    io_group.add_argument("--mrna-seq", type=str, default=None)
    io_group.add_argument("--csv", type=str, default=None)
    io_group.add_argument("--output", type=str, default=None)

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model-weights", type=str, required=True)
    model_group.add_argument("--params", type=str, default=None)

    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.params is not None:
        params_path = args.params
    else:
        params_path = Path(__file__).parent / "mRNA_param.json"
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = {**_DEFAULT_PARAMS, **json.load(f)}
    else:
        params = _DEFAULT_PARAMS.copy()
    params.setdefault("lr", 0.001)
    params.setdefault("loss", "mse")

    records = []
    if args.sirna_seq and args.mrna_seq:
        records.append(("input_siRNA", args.sirna_seq, "input_mRNA", args.mrna_seq))
    if args.csv:
        df_in = pd.read_csv(args.csv)
        for _, row in df_in.iterrows():
            records.append((
                row.get("siRNA", f"siRNA_{_}"),
                row["sirna_seq"],
                row.get("mRNA", f"mRNA_{_}"),
                row["mrna_seq"],
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
    print("Building graph...", file=sys.stderr)
    g = build_prediction_graph(records)

    generator = HinSAGENodeGenerator(
        g, batch_size=params["batch_size"],
        hop_samples=params["hop_samples"],
        head_node_type="interaction",
    )
    interaction_ids = list(g.nodes_of_type("interaction"))

    print("Building model...", file=sys.stderr)
    model = build_inference_model(generator, params)
    model.load_weights(args.model_weights)

    print("Predicting...", file=sys.stderr)
    test_interaction = pd.DataFrame(
        np.zeros((len(interaction_ids), 1)), index=interaction_ids)
    test_gen = generator.flow(test_interaction.index, test_interaction)
    preds = np.squeeze(model.predict(test_gen, verbose=0))

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
