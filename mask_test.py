import torch

input_ids = torch.arange(12).reshape(2, 6)
batch_size, seq_length = input_ids.shape

input_ids = torch.cat([input_ids, input_ids], dim=1)
seq_length *= 2

ar = torch.arange(seq_length, device=input_ids.device, dtype=input_ids.dtype)
segment_ids = ar % 3
print(segment_ids)

quasi_ids = torch.zeros_like(input_ids)
quasi_ids[:, :seq_length//2].fill_(1)
print(quasi_ids)

# batch_size = 5
# seq_length = 6
# segment_size = 2

# prefix_length = torch.tensor([1, 2, 4], dtype=torch.long)

# n_segments = seq_length // segment_size

# nw = torch.ones(batch_size, n_segments, n_segments, dtype=torch.bool)
# nw = torch.triu(nw, diagonal=1)
# nw = torch.repeat_interleave(nw, segment_size, dim=1)
# nw = torch.repeat_interleave(nw, segment_size, dim=2)

# sw = torch.ones(batch_size, n_segments, n_segments, dtype=torch.bool)
# sw = torch.triu(sw, diagonal=0)
# sw = torch.repeat_interleave(sw, segment_size, dim=1)
# sw = torch.repeat_interleave(sw, segment_size, dim=2)

# ne = torch.ones(batch_size, seq_length, seq_length, dtype=torch.bool)

# se = torch.ones(batch_size, seq_length, seq_length, dtype=torch.bool)
# se = torch.triu(se, diagonal=1)
# se = torch.logical_xor(se, ~sw)

# mask = torch.cat(
#     [
#         torch.cat([nw, ne], dim=2),
#         torch.cat([sw, se], dim=2)
#     ],
#     dim=1
# )

# out_mask = torch.zeros(batch_size, 2*seq_length, 2*seq_length, dtype=torch.float32)
# out_mask = torch.masked_fill(out_mask, mask, float('-inf'))

# print(out_mask)