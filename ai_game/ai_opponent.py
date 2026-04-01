"""Adaptive AI opponent using an ensemble ML model with confidence-weighted voting.

Ensemble: RandomForest + MLPClassifier + GaussianNB.

Prediction:
  Each model outputs predict_proba() over classes {1, 2, 3}.
  Each model's probability vector is weighted by its max confidence value.
  The weighted sum across models determines the final chosen action.

Logs per-model and ensemble confidences after every prediction.
"""

import os
import random
from typing import Tuple

import numpy as np

from .fighter import Fighter

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "game_logs.csv")


class AdaptiveAIOpponent(Fighter):
    """AI opponent using a confidence-weighted ensemble for move prediction."""

    CLASSES = [1, 2, 3]

    def __init__(self, name: str = "AI"):
        super().__init__(name, health=100, mp=50)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.naive_bayes import GaussianNB

        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        self.nb_model = GaussianNB()

        self._train_X: list = []
        self._train_y: list = []
        self._models_fitted: bool = False

        # Confidence logging (populated after each predict_move call)
        self.ensemble_confidence: float = 0.0
        self.last_confidences: dict = {"rf": 0.0, "nn": 0.0, "nb": 0.0}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def load_history(self):
        """Load past game_logs.csv to warm-up the ensemble."""
        import pandas as pd
        if not os.path.exists(LOG_FILE):
            return
        try:
            df = pd.read_csv(LOG_FILE)
            if "player_move" in df.columns and "ai_move" in df.columns and len(df) > 1:
                self._train_X = df[["player_move"]].values.tolist()
                self._train_y = df["ai_move"].values.tolist()
                self._fit_models()
        except Exception:
            pass

    def _fit_models(self):
        if len(self._train_X) < 2:
            return
        X = np.array(self._train_X)
        y = np.array(self._train_y)
        try:
            self.rf_model.fit(X, y)
            self.nn_model.fit(X, y)
            self.nb_model.fit(X, y)
            self._models_fitted = True
        except Exception:
            self._models_fitted = False

    def update_and_train(self, player_move: int, ai_move: int):
        """Add one data point and retrain models online."""
        self._train_X.append([player_move])
        self._train_y.append(ai_move)
        self._fit_models()

    # ------------------------------------------------------------------
    # Prediction — confidence-weighted voting
    # ------------------------------------------------------------------

    def predict_move(self, player_move: int) -> Tuple[int, float]:
        """
        Returns (chosen_move, ensemble_confidence_pct).

        Uses confidence-weighted voting:
          weight_i = max(predict_proba_i)
          final_proba = sum_i(weight_i * proba_i) / sum_i(weight_i)
        """
        if not self._models_fitted:
            chosen = random.choice(self.CLASSES)
            self.ensemble_confidence = 33.3
            self.last_confidences = {"rf": 33.3, "nn": 33.3, "nb": 33.3}
            return chosen, 33.3

        X_test = np.array([[player_move]])

        def safe_proba(model) -> np.ndarray:
            """Get probability vector aligned to CLASSES [1,2,3]."""
            uniform = np.ones(len(self.CLASSES)) / len(self.CLASSES)
            try:
                raw = model.predict_proba(X_test)[0]
                known = list(model.classes_)
                aligned = np.zeros(len(self.CLASSES))
                for idx, cls in enumerate(self.CLASSES):
                    if cls in known:
                        aligned[idx] = raw[known.index(cls)]
                if np.any(np.isnan(aligned)) or np.sum(aligned) == 0:
                    return uniform
                return aligned
            except Exception:
                return uniform

        rf_proba = safe_proba(self.rf_model)
        nn_proba = safe_proba(self.nn_model)
        nb_proba = safe_proba(self.nb_model)

        # Per-model confidence = max predicted probability
        rf_w = float(np.max(rf_proba))
        nn_w = float(np.max(nn_proba))
        nb_w = float(np.max(nb_proba))

        total_w = rf_w + nn_w + nb_w
        if total_w == 0.0:
            total_w = 1.0

        weighted = (rf_w * rf_proba + nn_w * nn_proba + nb_w * nb_proba) / total_w
        best_idx = int(np.argmax(weighted))
        chosen = self.CLASSES[best_idx]

        self.ensemble_confidence = round(float(np.max(weighted)) * 100, 2)
        self.last_confidences = {
            "rf": round(rf_w * 100, 2),
            "nn": round(nn_w * 100, 2),
            "nb": round(nb_w * 100, 2),
        }
        return chosen, self.ensemble_confidence
