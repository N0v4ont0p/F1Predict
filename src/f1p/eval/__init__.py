"""Evaluation harness (plan §10): rolling-origin forecasts, proper scores, baselines, CIs.

Every model — baselines today, the world model and its challengers later — is scored the same
way: for each event and checkpoint, it sees only data that was public at that moment
(``observed_at <= as_of``), produces a :class:`~f1p.eval.forecast.Forecast`, and is scored
against what actually happened.
"""
