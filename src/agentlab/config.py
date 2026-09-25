"""One place for every number a learner might want to change.

Model names and prices drift. Check them against the current OpenAI docs
before the session and edit them here — nothing else hard-codes them.
"""

# The model under test. Every notebook uses the same model, temperature and
# seed, so a change in any score comes from changes to the system around it
# rather than from swapping the model.
MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4.1-nano"      # used by the fallback demo in Notebook 05
JUDGE_MODEL = "gpt-4o-mini"          # LLM-as-judge (Notebook 01)
SIMULATOR_MODEL = "gpt-4o-mini"      # simulated customers (Notebook 01)
TEMPERATURE = 0
SEED = 7

# USD per 1M tokens (input, output). Illustrative — verify before quoting.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
}
DEFAULT_PRICE = (0.50, 1.50)

# A hard cap per notebook run. The meter refuses the next call once the
# notebook has spent this much, so a runaway loop can't run up the bill.
SPEND_LIMIT_USD = 1.00

# Agent loop defaults.
MAX_STEPS = 10

# Business rules for Acme Outfitters (the running case).
REFUND_WINDOW_DAYS = 30
REFUND_HUMAN_APPROVAL_OVER = 500.00
