"""
Run this ONCE before starting your app:
    python setup_map.py
Downloads TomTom Maps SDK + Leaflet into your static folder.
"""
import urllib.request, os, sys

static = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static, exist_ok=True)

files = {
    "leaflet.js":       "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    "leaflet.css":      "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
    "marker-icon.png":  "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    "marker-shadow.png":"https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
}

print("Downloading map files into /static ...\n")
for fname, url in files.items():
    dest = os.path.join(static, fname)
    if os.path.exists(dest):
        print(f"  ✅ already exists: {fname}")
        continue
    print(f"  downloading {fname}...", end=" ", flush=True)
    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=15).read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"✅ ({len(data):,} bytes)")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

print("\n✅ All map files ready!")
print("Now run: python app_final.py")