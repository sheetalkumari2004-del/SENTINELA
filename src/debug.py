from pathlib import Path
import sys
sys.path.insert(0, 'src')

_here = Path('src/dashboard.py').resolve().parent
ROOT = _here.parent if _here.name == 'src' else _here
DATA_CENTRALITY = str(ROOT / 'data' / 'processed' / 'astram_with_centrality.csv')

print('DATA_CENTRALITY:', DATA_CENTRALITY)

from data_prep import load_and_split
train, val, test = load_and_split(centrality_path=DATA_CENTRALITY)
print('SUCCESS:', len(train), len(val), len(test))