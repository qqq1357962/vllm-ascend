# PivotCoarseSelect

`torch.ops._C_ascend.npu_indexer_coarse_screen` takes `q_bar`, `w_bar`, and paged `key`. The reference-style keyword arguments are `actual_seq_lengths_query`, `actual_seq_lengths_key`, `block_table`, `coarse_count`, `has_window`, `group_size`, and `proxy_mode`. Set `coarse_count` to 2048 or 4096 and `has_window` to 1.

| Mode | Input metadata | Output |
| --- | --- | --- |
| `is_modify_localwindow=False` | One row per group. Query lengths are cumulative raw query ends. KV lengths are visible at each group’s final query. `proxy_mode=0` supplies one precomputed query and weight row per group. `proxy_mode=1` supplies raw query and weight rows. | One candidate row and one effective length per group. |
| `is_modify_localwindow=True` | One row per request. Query lengths are cumulative raw request ends. KV lengths are visible at the final query in each request. Supply raw queries and weights with `proxy_mode=1`, 32 query heads, and `group_size=4`. | One candidate row and one effective length per raw token. For a four-token group, the last three tokens replace the final one, two, and three positions with newly visible keys. |

The grouped route reserves up to `2 * group_size - 1` positions for the local window and ranks the historical prefix. Any window key already selected in history appears once. `aslk` gives the valid prefix length of each returned row. Unused positions contain `-1`.

The existing `npu_pivot_lightning_indexer` entry point dispatches to the same backend with the original C Pivot reuse geometry.
