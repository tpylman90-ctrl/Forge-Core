# Run matched tutor-v2 comparison

Run `.github/workflows/matched-tutor-v2.yml` on main.

For language, Blender, and Godot, compare three arms from the same verified Forge Seed parent and lane seed:
- control: original lane stream only
- direct-v2: worked explanation, procedure, reusable principle, same-case check
- tutor-v2: misconception diagnosis, Forge-initiated Ask-Why, short explanation/analogy, faded retrieval, changed transfer case, observable check

Keep all other settings and the 500-update budget matched, evaluate every 100 updates, preserve each selected checkpoint and parent/data hashes, and compare held-out direct-v2/tutor-v2 text plus lane/shared-language validation. Do not merge or promote from text loss alone. All lanes remain separate.
