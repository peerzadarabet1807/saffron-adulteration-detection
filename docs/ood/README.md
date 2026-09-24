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

These photos keep their own licenses; the repository's MIT license does not apply to them.
