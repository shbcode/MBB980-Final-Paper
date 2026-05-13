"""Train a SentencePiece BPE or Unigram model on a text corpus.

Usage:
    python -m src.tokenizers.train_sentencepiece \
        --input data/processed/bilingual.txt \
        --model_prefix models/sp_bpe \
        --vocab_size 32000 \
        --model_type bpe
"""

from __future__ import annotations

import click


@click.command()
@click.option("--input", "input_path", required=True, help="One sentence per line.")
@click.option("--model_prefix", required=True, help="Output prefix; writes .model and .vocab.")
@click.option("--vocab_size", default=32000, type=int)
@click.option("--model_type", default="bpe", type=click.Choice(["bpe", "unigram"]))
@click.option("--character_coverage", default=0.9995, type=float,
              help="Use 0.9995 for languages with large character sets like Chinese.")
def main(
    input_path: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
) -> None:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        normalization_rule_name="nmt_nfkc",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=8192,
    )
    click.echo(f"Wrote {model_prefix}.model and {model_prefix}.vocab")


if __name__ == "__main__":
    main()
