
import numpy as np


def get_shard(
    tokenizer,
    dataset,
    bs,
    max_length,
    num_tokens,
    b_factor=64,
    l_factor=128,
    min_thresh=0.5,
    max_thresh=1.5,
    verbose=True
):
    """
    
    Args:
        tokenizer: tokenizer object
        dataset: dataset object
        bs: batch size in tokens
        max_length: max sequence length
        num_tokens: total number of tokens in shard
        b_factor: batch size must be divisible by this factor
        l_factor: sequence length must be divisible by this factor
        min_thresh: minimum threshold as a proportion of bs
        max_thresh: maximum threshold as a proportion of bs
        verbose: verbosity flag
    """

    # accumulate text until we have enough tokens
    data = []
    total = 0
    for x in dataset:

        # estimated tokens after we pad to l_factor
        count = l_factor * np.ceil(x["token_count"] / l_factor)

        # save data as text, est_tokens
        data.append((x["text"], count))
        total += count

        # stop when we haave enough
        if total >= num_tokens:
            break

    # sort least to most tokens
    data = sorted(data, key=lambda d: d[1])

    # put data into batches of tokens
    data_out = []
    batch = []
    batch_counts = []
    for ind in range(len(data)):

        # add elem to batch
        text, count = data[ind]
        batch.append(text)
        batch_counts.append(count)

        # estimate tokens in batch, accounting for b_factor
        b_count = sum(batch_counts[:b_factor * (len(batch)//b_factor)])

        # process batch when big enough
        if b_count >= bs:

            # account for b_factor
            if len(batch) >= b_factor:
                batch = batch[:b_factor * (len(batch)//b_factor)]

                batch = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np",
                ).input_ids.astype(np.uint16)

                # account for l_factor
                if batch.shape[1] % l_factor != 0:
                    batch = np.pad(batch, ((0, 0), (0, l_factor - batch.shape[1] % l_factor)))

                # make sure our estimate wasn't too far off
                if batch.size > bs*min_thresh and batch.size < bs*max_thresh:
                    data_out.append(batch)

            # reset
            batch = []
            batch_counts = []

    if verbose:
        print(
            f"""
            Relative Shard Size: {sum([b.size for b in data_out]) / num_tokens:.3f}
            Token Density: {sum([(b!=tokenizer.pad_token_id).sum() for b in data_out]) / sum([b.size for b in data_out]):.3f}
            """
        )
    return data_out