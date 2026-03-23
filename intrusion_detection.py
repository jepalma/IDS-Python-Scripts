# intrusion_detection.py
# Rule-Based Expert System for Intrusion Detection
# Forward Chaining Inference Mechanism

def intrusion_detection():
    print("=== Intrusion Detection Expert System ===\n")

    # Collect facts from user
    failed_attempts = int(input("Enter failed login attempts: "))
    login_success = input("Login success after failures? (yes/no): ").lower()
    access_time = input("Enter access time (day/night): ").lower()
    geolocation = input("Enter geolocation (same/new): ").lower()
    device_status = input("Enter device status (trusted/unrecognized): ").lower()

    # Knowledge base (facts inferred)
    facts = {
        "brute_force": False,
        "suspicious_login": False,
        "unauthorized_device": False
    }

    severity = "NORMAL"
    actions = []

    # --------------------
    # Forward Chaining Rules
    # --------------------

    # Rule 1: Brute Force Attack
    if failed_attempts > 5 and login_success == "yes":
        facts["brute_force"] = True
        severity = "HIGH"
        actions.append("Trigger Security Alert")

    # Rule 2: Suspicious Login Activity
    if access_time == "night" and geolocation == "new":
        facts["suspicious_login"] = True
        if severity != "HIGH":
            severity = "MEDIUM"
        actions.append("Notify Security Team")

    # Rule 3: Unauthorized Device
    if device_status == "unrecognized":
        facts["unauthorized_device"] = True
        if severity == "NORMAL":
            severity = "LOW"
        actions.append("Require Multi-Factor Authentication (MFA)")

    # Rule 4: Escalation
    if facts["brute_force"] and facts["suspicious_login"]:
        severity = "HIGH"
        if "Trigger Security Alert" not in actions:
            actions.append("Trigger Security Alert")

    # --------------------
    # Output Results
    # --------------------
    print("\n=== Analysis Result ===")
    print(f"[ALERT] {severity} SEVERITY Intrusion Detected!\n")

    print("Indicators:")
    print(f"- Brute Force Attack: {'Yes' if facts['brute_force'] else 'No'}")
    print(f"- Suspicious Login: {'Yes' if facts['suspicious_login'] else 'No'}")
    print(f"- Unauthorized Device: {'Yes' if facts['unauthorized_device'] else 'No'}")

    print("\nSecurity Actions:")
    if actions:
        for action in set(actions):
            print(f"- {action}")
    else:
        print("- No action required (Normal activity)")

# Run system
if __name__ == "__main__":
    intrusion_detection()
