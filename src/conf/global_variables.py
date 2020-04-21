from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODELS_PATH = PROJECT_ROOT / 'models'

SEED = 42
# LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# TEXT_COLUMN = 'comment_text'


LABELS = ['label_1', 'label_2']
TEXT_COLUMN = 'text'