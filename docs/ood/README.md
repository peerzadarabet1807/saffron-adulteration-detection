# Non-saffron test photos

The classifier has no "not saffron" class, so it always answers with one of its five classes. To see how
it behaves on photos it was never meant for, it was run on seven openly licensed photos of red thread, yarn
and fabric from Wikimedia Commons. Predictions come from
[`scripts/evaluate_ood.py`](../../scripts/evaluate_ood.py) and are stored in
[`predictions.json`](predictions.json); photo 1 is the one in `docs/screenshots/05-low-confidence.png`.

| # | Photo | Prediction (top class) | Verdict | Low-confidence warning |
|---|---|---|---|---|
| 1 | Red thread on a spool | C3, 25.8% | counterfeit | yes |
| 2 | Tangle of red thread | C3, **75.4%** | counterfeit | **no** |
| 3 | White thread through a needle | C2, 30.4% | counterfeit | yes |
| 4 | Wooden spool of dark red thread | C3, 34.1% | counterfeit | yes |
| 5 | Red wool fabric texture | C1, 52.5% | **pure** | yes |
| 6 | Hand-spun red yarn | C2, 34.9% | pure | yes |
| 7 | Red backstitch spiral on white linen | C3, **82.9%** | counterfeit | **no** |

The warning fires for 5 of 7, but the two red-thread-on-light-background photos (2 and 7), the closest in
appearance to the training data, are classified with high confidence. The warning reduces, but does not
remove, the risk of trusting an answer about something that is not saffron.

The [input check](../../README.md#input-check), which now runs before the classifier, turns all seven of
these photos away (scores 0.5749 to 0.7740 against a threshold of 0.9419, [`input_check.json`](input_check.json)).

## The full set: 222 photos for the input check

To build and test the input check, the set was extended to 222 openly licensed photos from Wikimedia Commons
(CC0, public domain, CC BY, CC BY-SA; no NC or ND licences). [`build_ood_set.py`](../../scripts/build_ood_set.py)
found candidates with fixed search queries per category; the photos were then picked by hand from contact
sheets, before any input check was scored ([`selection.json`](selection.json), with a reason for every skipped
candidate), and downscaled to 640 px on the long side. They are in [`set/`](set), one folder per category.

- **Near** (100, including the original 7): look-alikes or things a user might photograph instead of one thread
  on paper: red thread and yarn, red fabric, chili threads, hair, saffron flowers, packaged saffron, food, small
  red objects. The check must reject these.
- **Grey zone** (40): look-alikes outside the five training classes that the task asked to measure and document
  rather than decide silently: safflower, corn silk, and saffron threads in a pile or dish rather than one
  thread on paper. They are reported separately from near and far.
- **Far** (82): people, documents, screenshots, scenes, everyday objects and other animals, the same Commons
  photos as the poultry repository's far categories.

[`split_ood.py`](../../scripts/split_ood.py) split the set once, before any tuning, into ood-tune (91) and
ood-test (131), stratified by category with a fixed seed ([`split.json`](split.json)); the original 7 photos are
all in ood-test.

On ood-test, per category: how often the classifier alone would give a confident answer (no low-confidence
warning), and how often the input check (`saffron-tinyclip-8m16-proto-1`, threshold 0.9419) lets the photo
through (from [`input_check.json`](input_check.json)):

| Category | Near, grey or far | Photos (tune / test) | Confident answer, classifier alone | Passes the input check |
|---|---|---|---|---|
| chili threads | near | 4 / 6 | 0 of 6 | 0 of 6 |
| food | near | 6 / 9 | 1 of 9 | 0 of 9 |
| hair | near | 4 / 6 | 0 of 6 | 0 of 6 |
| red fabric | near | 5 / 8 | 1 of 8 | 0 of 8 |
| red objects | near | 5 / 7 | 1 of 7 | 0 of 7 |
| red thread yarn | near | 8 / 11 | 2 of 11 | 0 of 11 |
| saffron flower | near | 4 / 7 | 1 of 7 | 0 of 7 |
| saffron packaged | near | 4 / 6 | 0 of 6 | 0 of 6 |
| corn silk | grey | 5 / 7 | 1 of 7 | 0 of 7 |
| safflower | grey | 5 / 7 | 2 of 7 | 0 of 7 |
| saffron loose | grey | 6 / 10 | 2 of 10 | 0 of 10 |
| documents | far | 6 / 8 | 2 of 8 | 0 of 8 |
| everyday objects | far | 6 / 8 | 0 of 8 | 0 of 8 |
| other animals | far | 6 / 8 | 0 of 8 | 0 of 8 |
| people | far | 5 / 7 | 0 of 7 | 0 of 7 |
| scenes | far | 6 / 8 | 1 of 8 | 0 of 8 |
| screenshots | far | 6 / 8 | 1 of 8 | 0 of 8 |
| **All** | | 91 / 131 | 15 of 131 | 0 of 131 |

No OOD photo in either split passes; the highest score is 0.8436 (a red pencil on paper). The grey zone is
therefore decided as "turned away" for the photos in this set, which show safflower and corn silk as flowers,
silks or heaps and saffron as piles. A single safflower petal or corn silk strand laid on paper like the
training photos was not available and is untested. Rates with 95% intervals, and the false-rejection rates on
real saffron-thread photos, are in [`results/input_check/evaluation.md`](../../results/input_check/evaluation.md).

## Credits

All photos were resized to 1024 px on the long side; no other changes.

| File | Author | License | Source |
|---|---|---|---|
| `red_thread_yarn.jpg` | 多多123 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Red_thread_(yarn).jpg) |
| `red_thread_kate_2112.jpg` | Kate Russell from Kent, United Kingdom | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Red_Thread_-_Flickr_-_Kate%5E2112.jpg) |
| `thread_amagill.jpg` | Andrew Magill from Boulder, USA | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Thread_-_Flickr_-_AMagill.jpg) |
| `spool_of_dark_red_sewing_thread_1.jpg` | W.carter | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Spool_of_dark_red_sewing_thread_1.jpg) |
| `red_wool_texture.jpg` | M0tty | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Red_wool_texture.jpg) |
| `handspun_yarn_16211767010.jpg` | distelfliege | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Handspun_yarn_(16211767010).jpg) |
| `backstitch_hand_embroidery.jpg` | TatjanaLT | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Backstitch_hand_embroidery.jpg) |

Credits for all 222 photos: [`CREDITS.md`](CREDITS.md) and [`credits.json`](credits.json). These photos keep their own licenses; the repository's MIT license does not apply to them.
