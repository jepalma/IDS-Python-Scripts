import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

#  1. LOAD & PREPARE DATA

CSV_FILE = "webLogin_Intrusion_Dataset.csv"

if not os.path.exists(CSV_FILE):
    print(f"[ERROR] '{CSV_FILE}' not found. Place it in the same folder as this script.")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)

print("=" * 55)
print("       INTRUSION DETECTION SYSTEM  (KNN)")
print("=" * 55)
print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Manual encoding to match user input choices exactly ──
#
# failed_login_attempts  → numeric, use as-is
# access_time            → day=0,  night=1
# geolocation            → same=0, new=1
# device_status          → trusted=0, unrecognized=1
# label (target)         → Non-Intrusion=0, Intrusion=1

df_enc = df.copy()
df_enc["access_time"]   = df_enc["access_time"].map({"day": 0, "night": 1})
df_enc["geolocation"]   = df_enc["geolocation"].map({"same": 0, "new": 1})
df_enc["device_status"] = df_enc["device_status"].map({"trusted": 0, "unrecognized": 1})
df_enc["label"]         = df_enc["label"].map({"Non-Intrusion": 0, "Intrusion": 1})

FEATURES = ["failed_login_attempts", "access_time", "geolocation", "device_status"]
TARGET   = "label"

X = df_enc[FEATURES].values
y = df_enc[TARGET].values

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)


#  2. TRAIN THE KNN MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"[INFO] Model trained successfully!")
print(f"[INFO] Test Accuracy : {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Intrusion", "Intrusion"]))

#  3. HELPER – validated integer input

def get_int(prompt, lo, hi):
    while True:
        try:
            val = int(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"    Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("    Invalid input. Please enter a whole number.")

#  4. INFINITE INPUT LOOP

print("=" * 55)
print("  Enter login attempt details  (Ctrl+C to exit)")
print("=" * 55)

while True:
    print()
    try:
        failed_logins = get_int(
            "Enter the number of failed login attempts: ", 0, 1000
        )
        access_time = get_int(
            "Select access time     (1 = Day,  2 = Night): ", 1, 2
        )
        geolocation = get_int(
            "Select geolocation     (1 = Same, 2 = New): ", 1, 2
        )
        device_status = get_int(
            "Select device status   (1 = Trusted, 2 = Unrecognized): ", 1, 2
        )

        # Map user choices (1/2) → encoded values (0/1)
        row = [
            failed_logins,
            access_time   - 1,   # Day→0, Night→1
            geolocation   - 1,   # Same→0, New→1
            device_status - 1,   # Trusted→0, Unrecognized→1
        ]

        row_scaled   = scaler.transform([row])
        prediction   = knn.predict(row_scaled)[0]
        proba        = knn.predict_proba(row_scaled)[0]
        confidence   = max(proba) * 100

        print("\n" + "─" * 45)
        if prediction == 1:
            print("  🚨  RESULT : *** INTRUSION DETECTED ***")
        else:
            print("  ✅  RESULT : Non-Intrusion  (Safe Login)")
        print(f"      Confidence : {confidence:.1f}%")
        print("─" * 45)

    except KeyboardInterrupt:
        print("\n\n[INFO] Exiting IDS. Goodbye!")
        break
