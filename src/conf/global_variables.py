from pathlib import Path

TOXIC_COMMENT_DATA_PATH = Path().cwd().parent / 'data'
SEED = 42
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
COMMENT = 'comment_text'
