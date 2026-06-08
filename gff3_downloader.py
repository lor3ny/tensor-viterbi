#!/usr/bin/env python3
"""
build_reference_gene.py
-----------------------
End-to-end script to build a per-nucleotide reference label vector for the
T2T-CHM13v2.0 chrY euchromatic MSY region (between PAR1 and Yq12 heterochromatin).

Output is a binary .gene file: one uint8 per nucleotide, where
    0 = intergenic
    1 = intron
    2 = exon

The output is aligned position-by-position with the FASTA slice you fetched
from the UCSC sequence API with start=2458320, end=26673214 on hs1 chrY.

Usage:
    python build_reference_gene.py                    # download + build
    python build_reference_gene.py --gff existing.gff3  # skip download
    python build_reference_gene.py --source refseq    # use RefSeq instead
"""

import argparse
import gzip
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Region definition — matches your FASTA slice exactly.
# UCSC sequence API uses 0-based half-open [start, end).
# ---------------------------------------------------------------------------
REGION_CHROM = "chrY"
REGION_START = 2_458_320       # end of PAR1
REGION_END   = 26_673_214      # start of Yq12 heterochromatin
REGION_LEN   = REGION_END - REGION_START  # 24,214,894

# ---------------------------------------------------------------------------
# Annotation sources.
# These URLs point to T2T-CHM13v2.0 GFF3 files. If a URL goes stale, browse
#   https://hgdownload.soe.ucsc.edu/hubs/GCA/009/914/755/GCA_009914755.4/
# and update accordingly.
# ---------------------------------------------------------------------------
ANNOTATION_SOURCES = {
    "catliftoff": {
        "url": "https://hgdownload.soe.ucsc.edu/hubs/GCA/009/914/755/GCA_009914755.4/genes/catLiftOffGenesV1.gff3.gz",
        "filename": "catLiftOffGenesV1.gff3.gz",
        "description": "CAT/Liftoff v1 — T2T Consortium standard annotation",
    },
    "refseq": {
        "url": "https://hgdownload.soe.ucsc.edu/hubs/GCA/009/914/755/GCA_009914755.4/genes/ncbiRefSeq.gff3.gz",
        "filename": "ncbiRefSeq.gff3.gz",
        "description": "NCBI RefSeq lifted onto T2T-CHM13v2.0",
    },
}


def download(url: str, dest: Path) -> None:
    """Download a URL to dest, showing simple progress."""
    if dest.exists():
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return
    print(f"  [get ] {url}")
    print(f"         -> {dest}")
    try:
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                print(f"         {total / 1e6:8.2f} MB", end="\r")
        print(f"         {total / 1e6:8.2f} MB  done.")
    except Exception as e:
        print(f"\nERROR downloading {url}: {e}", file=sys.stderr)
        print("\nIf your network blocks UCSC, download the file manually from:", file=sys.stderr)
        print(f"  {url}", file=sys.stderr)
        print("then re-run with: --gff <path-to-file>", file=sys.stderr)
        sys.exit(1)


def open_gff(path: Path):
    """Open a GFF3 file, transparently handling .gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def build_labels(gff_path: Path,
                 use_cds: bool = False,
                 verbose: bool = True) -> np.ndarray:
    """
    Walk the GFF3 file and paint a uint8 label vector of length REGION_LEN.

    Painting order matters:
        - first pass: features that span whole genes -> mark as intron (1)
        - second pass: exon/CDS features              -> overwrite as exon (2)
    Anything left untouched stays 0 (intergenic).

    If use_cds=True, only CDS features count as "exon" — this restricts the
    exon class to protein-coding regions only (no UTRs).
    """
    labels = np.zeros(REGION_LEN, dtype=np.uint8)

    # Counters for the sanity-check report
    n_gene_lines = 0
    n_exon_lines = 0
    n_cds_lines  = 0
    n_outside    = 0
    n_other_chrom = 0

    gene_like = {"gene", "pseudogene", "ncRNA_gene"}
    exon_like = {"CDS"} if use_cds else {"exon"}

    # Two passes so paint order is guaranteed regardless of GFF3 ordering.
    for pass_num, accept in enumerate([gene_like, exon_like], start=1):
        with open_gff(gff_path) as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9:
                    continue
                chrom = fields[0]
                ftype = fields[2]
                if chrom != REGION_CHROM:
                    if pass_num == 1:
                        n_other_chrom += 1
                    continue
                if ftype not in accept:
                    continue

                # GFF3 is 1-based inclusive on both ends.
                gff_start = int(fields[3])
                gff_end   = int(fields[4])

                # Convert to 0-based half-open absolute coords on chrY...
                abs_s = gff_start - 1
                abs_e = gff_end
                # ...then shift into local coords of our slice.
                s = abs_s - REGION_START
                e = abs_e - REGION_START
                # Clip to region.
                s_clipped = max(s, 0)
                e_clipped = min(e, REGION_LEN)
                if s_clipped >= e_clipped:
                    if pass_num == 1:
                        n_outside += 1
                    continue

                if pass_num == 1:
                    labels[s_clipped:e_clipped] = 1
                    n_gene_lines += 1
                else:
                    if use_cds:
                        labels[s_clipped:e_clipped] = 2
                        n_cds_lines += 1
                    else:
                        labels[s_clipped:e_clipped] = 2
                        n_exon_lines += 1

    if verbose:
        print(f"  GFF features used:")
        print(f"    gene-like features painted as intron:  {n_gene_lines:,}")
        if use_cds:
            print(f"    CDS features painted as exon:        {n_cds_lines:,}")
        else:
            print(f"    exon features painted as exon:       {n_exon_lines:,}")
        print(f"    chrY features fully outside region:    {n_outside:,}")
        print(f"    lines on other chromosomes (skipped):  {n_other_chrom:,}")

    return labels


def report(labels: np.ndarray) -> None:
    """Print class distribution as a sanity check."""
    n = labels.size
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    n2 = int((labels == 2).sum())
    print()
    print(f"  Reference label distribution over {n:,} bp:")
    print(f"    intergenic (0): {n0:>12,}  {n0/n:7.3%}")
    print(f"    intron     (1): {n1:>12,}  {n1/n:7.3%}")
    print(f"    exon       (2): {n2:>12,}  {n2/n:7.3%}")
    print()
    print("  Expected ballpark for euchromatic MSY:")
    print("    intergenic ~85–95%, intron ~5–15%, exon <1%.")
    print("  If exon is 0% or intergenic is 100%, something is wrong with parsing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(ANNOTATION_SOURCES.keys()),
                    default="catliftoff",
                    help="Which annotation track to use (default: catliftoff)")
    ap.add_argument("--gff", type=Path, default=None,
                    help="Use an existing local GFF3 (skip download)")
    ap.add_argument("--out", type=Path, default=Path("chrY_msy_reference.gene"),
                    help="Output .gene file (default: chrY_msy_reference.gene)")
    ap.add_argument("--workdir", type=Path, default=Path("./annot_cache"),
                    help="Where to cache downloaded files")
    ap.add_argument("--use-cds", action="store_true",
                    help="Label only CDS regions as exon (no UTRs). "
                         "Use this if your HSMM exon class means protein-coding.")
    args = ap.parse_args()

    print(f"Region: {REGION_CHROM}:{REGION_START}-{REGION_END} "
          f"(length {REGION_LEN:,} bp)")
    print(f"Output: {args.out}")
    print()

    # ---- get the GFF3 ----
    if args.gff is not None:
        gff_path = args.gff
        if not gff_path.exists():
            print(f"ERROR: --gff {gff_path} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Using existing GFF3: {gff_path}")
    else:
        src = ANNOTATION_SOURCES[args.source]
        print(f"Annotation source: {args.source} — {src['description']}")
        args.workdir.mkdir(parents=True, exist_ok=True)
        gff_path = args.workdir / src["filename"]
        download(src["url"], gff_path)

    print()
    print("Building label vector...")
    labels = build_labels(gff_path, use_cds=args.use_cds)
    report(labels)

    # ---- write output ----
    line_width = 30
    text = "".join(str(v) for v in labels)
    with open(args.out, "w") as f:
        f.write(f"# Gene structure predictions | source: {args.out.stem}.fa | generated by tensor-viterbi\n")
        for i in range(0, len(text), line_width):
            f.write(text[i:i + line_width] + "\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes, "
          f"{labels.size:,} nucleotides)")


if __name__ == "__main__":
    main()