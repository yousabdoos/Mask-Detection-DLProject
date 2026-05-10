import requests
import os
import json

# ======================
# SETTINGS
# ======================
API_URL = "http://127.0.0.1:8000/predict"
TEST_IMAGES_DIR = "Real-World Testing data"

# ======================
# HELPERS
# ======================
def print_separator():
    print("-" * 50)

def test_image(image_path):

    print(f"\nTesting: {image_path}")
    print_separator()

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        result = response.json()

        if "error" in result:
            print(f"  API Error    : {result['error']}")
        else:
            print(f"  Status       : {result['status']}")
            print(f"  Action       : {result['action']}")
            print(f"  Confidence   : {result['confidence']:.4f}")
            print(f"  Probabilities: {json.dumps(result['probabilities'], indent=4)}")

    else:
        print(f"  HTTP Error: {response.status_code}")

# ======================
# MAIN
# ======================
if __name__ == "__main__":

    print("=" * 50)
    print("   Face Mask Detection API - Test Script")
    print("=" * 50)

    # ── Check API is running ──
    try:
        health = requests.get("http://127.0.0.1:8000")
        print("\n[OK] API is running.\n")
    except requests.exceptions.ConnectionError:
        print("\n[FAIL] Could not connect to API.")
        print("   Make sure the server is running:")
        print("   uvicorn main:app --reload --port 8000")
        exit(1)

    # ── Find test images ──
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"[FAIL] Folder '{TEST_IMAGES_DIR}' not found.")
        print(f"   Create it and put some .jpg/.png images inside.")
        exit(1)

    image_files = [
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"[FAIL] No images found in '{TEST_IMAGES_DIR}' folder.")
        exit(1)

    print(f"Found {len(image_files)} image(s) to test.\n")

    # ── Test each image ──
    passed = 0
    failed = 0

    for image_path in image_files:
        try:
            test_image(image_path)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    # ── Summary ──
    print("\n" + "=" * 50)
    print(f"   Results: {passed} passed  |  {failed} failed")
    print("=" * 50)