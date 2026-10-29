import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "spam-sms-detection")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from gui_app import SpamDetectorGUI

if __name__ == "__main__":
    app = SpamDetectorGUI()
    app.mainloop()
