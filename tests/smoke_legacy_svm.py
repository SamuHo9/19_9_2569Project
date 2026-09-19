"""Exercise a legacy SVM's actual sklearn training with graphics stubbed out.

Produces real predictions in a fresh run; does not claim plot rendering was tested.
"""
import ast
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Model'))
from data_contract import load_training_data, PROTOCOL

class NoGraphics:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

path = ROOT / 'Model/Ds004469/left/SVM/train_svm_pls.py'
tree = ast.parse(path.read_text(encoding='utf-8'))
selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
ns = dict(globals(), __file__=str(path), plt=NoGraphics(), sns=NoGraphics())
exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)
ns['run_pipeline']()
manifest = Path('run_manifest.json')
data = json.loads(manifest.read_text(encoding='utf-8'))
data['verification_mode'] = 'real_sklearn_training; plotting_calls_stubbed'
manifest.write_text(json.dumps(data, indent=2), encoding='utf-8')
print('SMOKE_OUTPUT:', Path.cwd())
