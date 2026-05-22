# Terminal 1 — FastAPI backend (from project root)
npm

# Terminal 2 — Frontend dashboard
npm run dev

# Terminal 3 — Upload video to API (from src/capture/)
# python register_camera.py --write-config   # once, while API is up
# python capture.py                        # POST video → /api/detect (pipeline server-side)
# python test_detect_api.py --max-triggers 5
# python capture.py --local                # run VideoCapturer + LPR without HTTP