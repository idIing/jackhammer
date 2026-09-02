# Adding seed datasets

The frozen v1 battery answers one question: relative agent performance on its published IID seeds.
Different questions need explicit datasets, not edits to that battery.

A manifest uses this shape:

```json
{
  "schema": "jackhammer.seed-dataset/v1",
  "name": "joker-access-coverage",
  "version": 1,
  "description": "Seeds sampled to study early Joker access; not IID performance.",
  "splits": {
    "sample": ["PVRQ4K5A", "4NNGD2DN"]
  },
  "metadata": {"sampling": "document the selection procedure here"}
}
```

Every seed is an eight-character Balatro seed drawn from
`123456789ABCDEFGHIJKLMNPQRSTUVWXYZ`. Each split must be non-empty and duplicate-free. Different
splits may overlap so coverage levels can be nested; document that relationship in `metadata`. The
loader stamps the manifest's raw-byte digest, path, name, version, description, metadata, and
selected split.

Run it with:

```bash
uv run python scripts/evaluate.py \
  --agent my-agent-v1 --vs greedy-shop \
  --dataset path/to/dataset.json --split sample
```

Artifacts from this route use protocol `jackhammer/dataset-eval/v1` and dataset scope `diagnostic`.
They cannot be paired with `jackhammer/v1` artifacts.

## Coverage studies

For questions like “how does seed coverage affect measured agent strength?”, add immutable manifests
for the pre-declared coverage levels or sampling rounds. Keep the agent pair, simulator pin, and
metric fixed; report each manifest digest and the overlap or nesting relationship between levels.
Do not interpret a deliberately difficult or feature-enriched sample as ordinary-distribution
performance.

Commit a dataset only when its generation or selection procedure is documented. Large generated
datasets may live in a release asset or external archive, but the manifest and content digest must
remain versioned here.
