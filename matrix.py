import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -------------------------
# 🔹 Load your game logs CSV
# -------------------------
df = pd.read_csv(r"C:\Users\Vikra\OneDrive\Desktop\ai_game\logs\game_logs.csv")


# Optional sanity checks
print("PlayerMove counts:\n", df['PlayerMove'].value_counts())
print("AIMove counts:\n", df['AIMove'].value_counts())

# -------------------------------
# 🔹 PART 1: Train a Model to Predict PlayerMove
# -------------------------------
print("\n=== MODEL TRAINING: Predicting PlayerMove from Game Stats ===")

# Features & Target
features = ['AIMove', 'AIDamage', 'AIMPUsed', 'PlayerDamage', 'PlayerMPUsed']
target = 'PlayerMove'

X = df[features]
y = df[target]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Confusion Matrix
cm_model = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_model, display_labels=[1, 2, 3])
disp.plot(cmap='Greens')
plt.title("🔍 Model Prediction: PlayerMove (True) vs Predicted")
plt.show()

# Classification Report
print("\nClassification Report (Model Prediction):")
print(classification_report(y_test, y_pred, labels=[1, 2, 3]))

# -----------------------------------
# 🔹 PART 2: Analyze AI Move vs Player Move (In-Game)
# -----------------------------------
print("\n=== GAME ANALYSIS: Comparing AI's Moves to Player's Actual Moves ===")

# Ground truth and AI's chosen move
y_true_game = df['PlayerMove']
y_pred_game = df['AIMove']

# Confusion Matrix
cm_game = confusion_matrix(y_true_game, y_pred_game, labels=[1, 2, 3])
print("Confusion Matrix (Player vs AI Moves):\n", cm_game)

# Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_game, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
plt.xlabel("AI Move")
plt.ylabel("Player Move (Actual)")
plt.title("🎮 In-Game Move Analysis: Player vs AI")
plt.show()

# Classification Report
print("\nClassification Report (AI vs Player Moves):")
print(classification_report(y_true_game, y_pred_game, labels=[1, 2, 3]))
