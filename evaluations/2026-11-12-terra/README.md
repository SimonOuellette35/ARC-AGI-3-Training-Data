# Terra evaluation of the homemade games

Recorded 12 November 2026. Contributed by 82deutschmark with GPT-6.1 Sol. This is independent model-play evaluation evidence, not generated training demonstrations or changes to the games.

## Outcome

One attempt on each of the **22 homemade games**: **14 fully solved; 129/153 levels cleared**. The final sixteen-game batch accounts for 12 solved games and 104/112 levels. These are observed outcomes under a time budget, not an estimate of general model success probability.

| Game | Run alias | Levels | Actions | Ending | Evidence |
|---|---|---|---|---|---|
| frog_crossing | SO01 | 6/7 | 222 | Inference timeout | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so01&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so01.jsonl.gz) |
| gem_mine | SO02 | 4/7 | 855 | Inference timeout | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so02&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so02.jsonl.gz) |
| aquarium_sorter | SO03 | 1/7 | 157 | Inference timeout | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so03&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so03.jsonl.gz) |
| balloon_festival | SO04 | 1/7 | 341 | Inference timeout | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so04&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so04.jsonl.gz) |
| dungeon_key_hunt | SO05 | 7/7 | 1002 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so05&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so05.jsonl.gz) |
| color_sorter | SO06 | 6/6 | 70 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so06&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so06.jsonl.gz) |
| badge_placement | SO07 | 7/7 | 426 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so07&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so07.jsonl.gz) |
| ball_sifting | SO08 | 6/6 | 537 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so08&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so08.jsonl.gz) |
| flood | SO09 | 7/7 | 440 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so09&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so09.jsonl.gz) |
| ice_rink_stars | SO10 | 7/7 | 137 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so10&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so10.jsonl.gz) |
| klotski | SO11 | 7/7 | 121 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so11&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so11.jsonl.gz) |
| maze | SO12 | 7/7 | 794 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so12&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so12.jsonl.gz) |
| one_step_tetris | SO13 | 8/8 | 77 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so13&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so13.jsonl.gz) |
| paired_pieces | SO14 | 4/7 | 119 | Inference timeout | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so14&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so14.jsonl.gz) |
| potion_brewer | SO15 | 6/7 | 487 | Time limit | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so15&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so15.jsonl.gz) |
| projectile_dodge | SO16 | 7/7 | 189 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so16&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so16.jsonl.gz) |
| rail_selector | SO17 | 7/7 | 128 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so17&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so17.jsonl.gz) |
| sheep_herder | SO18 | 4/7 | 479 | Inference timeout | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so18&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so18.jsonl.gz) |
| sokoban | SO19 | 7/7 | 341 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so19&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so19.jsonl.gz) |
| symmetric_maze | SO20 | 7/7 | 117 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so20&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so20.jsonl.gz) |
| tile_rotation | SO21 | 7/7 | 121 | Completed | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so21&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so21.jsonl.gz) |
| toy_train | SO22 | 6/7 | 1025 | Time limit | [inspect](https://82deutschmark.github.io/arc3-synthetic-games/replays/external-terra/viewer.html#run=so22&game=0&turn=1) · [recording](https://github.com/82deutschmark/arc3-synthetic-games/blob/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra/so22.jsonl.gz) |

## Settings and interpretation

- Model: `gpt-6.1-terra`, medium reasoning, through Codex with ChatGPT subscription authentication.
- Existing `daniel-main-1030-rulesonce` tool-agent harness; one attempt per game, 900-second game budget and 105-second inference-request timeout. No additional attempts were launched for this contribution.
- Opaque SO aliases replace only the environment class name and constructor game identifier. Original game names are restored in this report; original and staged source SHA-256 values are in `results.json`.
- No per-level human baselines were supplied. Stored normalized zero scores are therefore not meaningful; report levels/actions instead.
- “Inference timeout” and “Time limit” are budget/inference endings, not proof that a game is unsolvable or defective. Raw harness states such as `crashed` are preserved separately and must not be interpreted as an environment crash.
- Original recorder action labels sometimes disagreed with Terra's actual inputs. Published frame recordings were reconstructed from actual actions and checked against saved frames. The correction counts are retained per run. The Run Inspector contains recorded reasoning notes, Python calls, tool results, and frames.
- Do not treat these trajectories as training-ready: they include failed exploration, resets and incomplete games, and have not been converted or validated against this repository's demonstration schema.
- This package does not establish absence of prior model exposure to public game code. It is not a contamination-controlled benchmark or an official ARC-AGI score.

## Portable evidence

[`results.json`](results.json) contains every game's outcome, exact original and aliased source hashes, game source path, and immutable compressed frame/overview/game-record URLs. All source hashes match this repository's evaluated revision, `e7e0494826eefc2874af2731e3dcdf2ad615e8b0`. Every artifact URL is pinned to the publication commit rather than a moving branch. The browser inspector is a convenient live interface; use the pinned artifact URLs for durable evidence.

The last-batch runtime manifest records Python, package versions, and imported harness module hashes. It is explicitly scoped to SO07–SO22; earlier batches do not have this additional runtime snapshot. Credentials and machine-local absolute paths are excluded.

[Published results and previews](https://82deutschmark.github.io/arc3-synthetic-games/#results) · [Pinned complete replay directory](https://github.com/82deutschmark/arc3-synthetic-games/tree/dccc47108dc86d73dad6d125f0d10126c7c285eb/docs/replays/external-terra)

## Verification for this contribution

All 22 original source hashes were recomputed from the upstream revision, all aliases/names were unique, aggregate totals were recomputed from original saved result records, and all 66 pinned artifact paths were checked in the published Git tree. No game, solver, generator, adapter or dependency files were modified.
