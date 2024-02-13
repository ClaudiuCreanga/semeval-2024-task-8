import torch
from transformers import BatchEncoding, PreTrainedTokenizer

from lib.data.exceptions import InconsistentSplittingParamsException


def check_split_parameters_consistency(
    tokenizer: PreTrainedTokenizer,
    chunk_size: int,
    stride: int,
    min_chunk_size: int
) -> None:
    if chunk_size > tokenizer.model_max_length:
        raise InconsistentSplittingParamsException(
            f"Size of each chunk cannot be bigger than "
            f"model_max_length = {tokenizer.model_max_length}!"
        )
    if min_chunk_size > chunk_size:
        raise InconsistentSplittingParamsException(
            f"Minimal length ({min_chunk_size}) "
            f"cannot be bigger than size ({chunk_size})!"
        )
    if stride > chunk_size:
        raise InconsistentSplittingParamsException(
            f"Stride ({stride}) cannot be bigger than size ({{chunk_size}})! "
            f"Chunks must overlap or be near each other!"
        )


def split_overlapping(
    tokenizer: PreTrainedTokenizer,
    tensor: torch.Tensor,
    chunk_size: int,
    stride: int,
    min_chunk_size: int,
) -> [torch.Tensor]:
    check_split_parameters_consistency(
        tokenizer,
        chunk_size,
        stride,
        min_chunk_size,
    )

    result = [tensor[i : i + chunk_size] for i in range(0, len(tensor), stride)]
    if len(result) > 1:
        result = [x for x in result if len(x) >= min_chunk_size]
    return result


def split_encoding_into_smaller_chunks(
    tokenizer: PreTrainedTokenizer,
    encoding: BatchEncoding,
    chunk_size: int,
    stride: int,
    min_chunk_size: int,
) -> ([torch.Tensor], [torch.Tensor]):
    input_id_chunks = split_overlapping(
        tokenizer,
        encoding["input_ids"][0],
        chunk_size,
        stride,
        min_chunk_size
    )
    attention_mask_chunks = split_overlapping(
        tokenizer,
        encoding["attention_mask"][0],
        chunk_size,
        stride,
        min_chunk_size
    )

    return input_id_chunks, attention_mask_chunks


def add_special_tokens_at_beginning_and_end(
    tokenizer: PreTrainedTokenizer,
    input_id_chunks: [torch.Tensor],
    attention_mask_chunks: [torch.Tensor],
) -> None:
    if len(input_id_chunks) == 0:
        input_id_chunks.append(
            torch.cat([
                torch.tensor([tokenizer.cls_token_id]),
                torch.tensor([tokenizer.sep_token_id]),
            ])
        )
        attention_mask_chunks.append(
            torch.cat([
                torch.tensor([1]),
                torch.tensor([1])
            ])
        )

        return

    for chunk_id in range(len(input_id_chunks)):
        input_id_chunks[chunk_id] = torch.cat(
            [
                torch.tensor([tokenizer.cls_token_id]),
                input_id_chunks[chunk_id],
                torch.tensor([tokenizer.sep_token_id]),
            ]
        )
        attention_mask_chunks[chunk_id] = torch.cat(
            [torch.tensor([1]), attention_mask_chunks[chunk_id], torch.tensor([1])]
        )


def add_padding_tokens(
    tokenizer: PreTrainedTokenizer,
    input_id_chunks: [torch.Tensor],
    attention_mask_chunks: [torch.Tensor],
    max_len: int,
) -> None:
    for chunk_id in range(len(input_id_chunks)):
        padding_len = max_len - input_id_chunks[chunk_id].shape[0]

        if padding_len > 0:
            input_id_chunks[chunk_id] = torch.cat(
                [
                    input_id_chunks[chunk_id],
                    torch.tensor(
                        [tokenizer.pad_token_id] * padding_len
                    )
                ]
            )
            attention_mask_chunks[chunk_id] = torch.cat(
                [
                    attention_mask_chunks[chunk_id],
                    torch.tensor(
                        [0] * padding_len
                    )
                ]
            )


def stack_chunks(chunks: [torch.Tensor]) -> torch.Tensor:
    return torch.stack(chunks)


def tokenize_long_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    truncate_documents: bool,
    max_document_len: int,
    max_len: int,
    chunk_size: int,
    stride: int,
    min_chunk_size: int,
    debug: bool = False,
) -> [[str], torch.Tensor]:
    # Split text into tokens
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=truncate_documents,
        max_length=max_document_len if truncate_documents else None,
        return_token_type_ids=False,
        return_tensors="pt"
    )

    if debug:
        print(f"Total tokens: {encoding['input_ids'].shape[1]}")

    input_id_chunks, attention_mask_chunks = split_encoding_into_smaller_chunks(
        tokenizer=tokenizer,
        encoding=encoding,
        chunk_size=chunk_size,
        stride=stride,
        min_chunk_size=min_chunk_size,
    )

    add_special_tokens_at_beginning_and_end(
        tokenizer, input_id_chunks, attention_mask_chunks)
    add_padding_tokens(
        tokenizer, input_id_chunks, attention_mask_chunks, max_len
    )

    if debug:
        print(f"Number of chunks: {len(input_id_chunks)}")

        for chunk_idx, input_id_chunk in enumerate(input_id_chunks):
            print(f"Chunk {chunk_idx} number of tokens: {input_id_chunk.shape[0]}")

    input_ids = stack_chunks(input_id_chunks).long()
    attention_mask = stack_chunks(attention_mask_chunks).int()

    if debug:
        print(f"Input ids shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")

        print()
        print("#" * 25)
        print()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def tokenize_long_texts(
    texts: [str],
    tokenizer: PreTrainedTokenizer,
    truncate_documents: bool,
    max_document_len: int,
    max_len: int,
    chunk_size: int,
    stride: int,
    min_chunk_size: int,
    debug: bool = False,
) -> BatchEncoding:
    encodings = [
        tokenize_long_text(
            text=text,
            tokenizer=tokenizer,
            truncate_documents=truncate_documents,
            max_document_len=max_document_len,
            max_len=max_len,
            chunk_size=chunk_size,
            stride=stride,
            min_chunk_size=min_chunk_size,
            debug=debug,
        ) for text in texts
    ]

    final_input_ids = [e["input_ids"] for e in encodings]
    final_attention_mask = [e["attention_mask"] for e in encodings]

    if debug:
        print()
        print("-" * 50)
        print()

        print(f"Number of texts: {len(final_input_ids)}")
        print()

        for idx in range(len(final_input_ids)):
            print(f"Input ids shape: {final_input_ids[idx].shape}")
            print(f"Attention mask shape: {final_attention_mask[idx].shape}")

    return BatchEncoding({
        "input_ids": final_input_ids,
        "attention_mask": final_attention_mask
    })
