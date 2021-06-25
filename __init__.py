import os
import sys
from pathlib import Path

if 'MVT_ROOT' in os.environ:
    MVT_ROOT = os.getenv('MVT_ROOT')
    print('Get MVT_ROOT: ', MVT_ROOT)
else:
    MVT_ROOT = str(Path(__file__).absolute().parent)
    os.environ['MVT_ROOT'] = MVT_ROOT
    print('Set MVT_ROOT: ', MVT_ROOT)

sys.path.insert(0, MVT_ROOT)
print('Add {} to PYTHONPATH'.format(MVT_ROOT))
