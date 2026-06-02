# File info
- kenlm_predictions_test.csv: KenLM predictions from test.py
- test_predictions_kenlm.csv: KenLM predictions from kenlm_decoder.py (LM 1: leipzig corpus model)
- test_predictions_combined_kenlm.csv: KenLM predictions from kenlm_decoder.py (LM 2: leipzig corpus model + augmented LUMINA - combined corpus)
- beam_predictions.csv: Beam search only and optional + n-gram from test.py
- test_metrics.json: Generated from test.py
## Sweep file
To determine which parameters are best for the KenLM (beam width and alpha)