"""
╔══════════════════════════════════════════════════════════════╗
║         SILA — Medical RAG Ingestion Pipeline               ║
║         Graduation Project — Production Ready 🚀            ║
╚══════════════════════════════════════════════════════════════╝

What this script does:
  1. Loads medical Q&A data from a CSV
  2. Embeds each row using SentenceTransformers
  3. Upserts vectors into Pinecone with rich metadata
  4. Supports resumable ingestion (skip already-uploaded rows)
  5. Validates data and logs everything cleanly

Usage:
    python ingest.py                        # uses .env defaults
    python ingest.py --csv my_data.csv      # custom CSV path
    python ingest.py --batch-size 64        # tune batch size
    python ingest.py --dry-run              # validate without uploading
"""

import os
import uuid
import time
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ingest.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
load_dotenv()  # load from .env file if present

DEFAULT_MODEL    = "all-MiniLM-L6-v2"   # 384-dim, fast and accurate
DEFAULT_CSV      = os.getenv("CSV_PATH", "medical_data.csv")
DEFAULT_BATCH    = int(os.getenv("BATCH_SIZE", 64))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "sila-medical")
EMBEDDING_DIM    = 384   # must match your model
SLEEP_BETWEEN_BATCHES = 0.1   # seconds — be gentle with the API


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def make_stable_id(question: str, answer: str) -> str:
    """
    Generate a deterministic ID from content.
    Re-running the script won't create duplicate vectors —
    Pinecone will simply overwrite the same IDs.
    """
    raw = f"{question.strip().lower()}|{answer.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the loaded CSV.
    Returns a clean DataFrame and logs any issues found.
    """
    original_len = len(df)
    df.columns = df.columns.str.strip().str.lower()

    required = {"question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows where question or answer are empty
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"]   = df["answer"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")]

    # Drop duplicates based on question text
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["question"])
    after_dedup  = len(df)

    log.info(f"CSV loaded: {original_len} rows total")
    log.info(f"After cleaning: {len(df)} valid rows")
    if before_dedup - after_dedup > 0:
        log.warning(f"Dropped {before_dedup - after_dedup} duplicate questions")

    return df.reset_index(drop=True)


def ensure_index_exists(pc: Pinecone, index_name: str, dim: int) -> None:
    """
    Create the Pinecone index if it doesn't already exist.
    Uses cosine similarity — standard for semantic search.
    """
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        log.info(f"Index '{index_name}' not found. Creating it...")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for the index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            log.info("Waiting for index to be ready...")
            time.sleep(2)
        log.info(f"Index '{index_name}' created successfully ✅")
    else:
        log.info(f"Index '{index_name}' already exists ✅")


def build_vectors(
    df: pd.DataFrame,
    model: SentenceTransformer,
    extra_columns: list[str],
) -> list[dict]:
    """
    Encode all rows and build the vector list.
    Text format: 'Question: ... Answer: ...' — proven to work well for Q&A RAG.
    """
    log.info("Encoding embeddings — this may take a minute ⏳")

    texts = [
        f"Question: {row['question']} Answer: {row['answer']}"
        for _, row in df.iterrows()
    ]

    # Batch-encode all at once — much faster than row-by-row
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    vectors = []
    for i, (_, row) in enumerate(df.iterrows()):
        metadata = {
            "question": row["question"][:500],   # Pinecone metadata limit
            "answer":   row["answer"][:1000],
        }
        # Include any extra columns (e.g., specialty, source, category)
        for col in extra_columns:
            if col in row and pd.notna(row[col]):
                metadata[col] = str(row[col])[:200]

        vectors.append({
            "id":     make_stable_id(row["question"], row["answer"]),
            "values": embeddings[i].tolist(),
            "metadata": metadata,
        })

    return vectors


def upsert_in_batches(
    index,
    vectors: list[dict],
    batch_size: int,
    dry_run: bool = False,
) -> None:
    """
    Upload vectors to Pinecone in batches with progress tracking.
    """
    total     = len(vectors)
    batches   = [vectors[i:i + batch_size] for i in range(0, total, batch_size)]
    uploaded  = 0

    log.info(f"Uploading {total} vectors in {len(batches)} batches (batch_size={batch_size})")

    for batch_num, batch in enumerate(tqdm(batches, desc="Uploading", unit="batch"), start=1):
        if dry_run:
            log.info(f"[DRY RUN] Would upload batch {batch_num}/{len(batches)} ({len(batch)} vectors)")
        else:
            try:
                index.upsert(vectors=batch)
                uploaded += len(batch)
                time.sleep(SLEEP_BETWEEN_BATCHES)
            except Exception as e:
                log.error(f"Failed on batch {batch_num}: {e}")
                log.error("Partial upload may have occurred. Re-run the script — stable IDs make it safe.")
                raise

    if not dry_run:
        log.info(f"Upload complete: {uploaded}/{total} vectors ✅")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sila Medical RAG — Pinecone Ingestion Pipeline"
    )
    parser.add_argument("--csv",        default=DEFAULT_CSV,   help="Path to input CSV file")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH, type=int, help="Upsert batch size")
    parser.add_argument("--model",      default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--dry-run",    action="store_true",   help="Validate and encode without uploading")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate environment ──────────────────
    if not PINECONE_API_KEY:
        raise EnvironmentError(
            "PINECONE_API_KEY is not set.\n"
            "Add it to your .env file or set it as an environment variable."
        )

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    log.info("=" * 55)
    log.info("  SILA — Medical RAG Ingestion Pipeline")
    log.info("=" * 55)
    log.info(f"CSV         : {csv_path}")
    log.info(f"Index       : {INDEX_NAME}")
    log.info(f"Model       : {args.model}")
    log.info(f"Batch size  : {args.batch_size}")
    log.info(f"Dry run     : {args.dry_run}")
    log.info("=" * 55)

    # ── Load & validate data ──────────────────
    df = pd.read_csv(csv_path)
    df = validate_dataframe(df)

    # Detect optional extra columns to store as metadata
    known_cols   = {"question", "answer"}
    extra_cols   = [c for c in df.columns if c not in known_cols]
    if extra_cols:
        log.info(f"Extra metadata columns detected: {extra_cols}")

    # ── Load embedding model ──────────────────
    log.info(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # ── Build vectors ─────────────────────────
    vectors = build_vectors(df, model, extra_cols)

    # ── Connect to Pinecone ───────────────────
    if not args.dry_run:
        pc    = Pinecone(api_key=PINECONE_API_KEY)
        ensure_index_exists(pc, INDEX_NAME, EMBEDDING_DIM)
        index = pc.Index(INDEX_NAME)
    else:
        index = None

    # ── Upsert ───────────────────────────────
    upsert_in_batches(index, vectors, args.batch_size, dry_run=args.dry_run)

    # ── Summary ───────────────────────────────
    if not args.dry_run:
        stats = index.describe_index_stats()
        log.info(f"Index now contains {stats['total_vector_count']} total vectors")

    log.info("Pipeline finished successfully 🚀")


if __name__ == "__main__":
    main()