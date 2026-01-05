import sys
import os

# Tambahkan path project ke sys.path
path = '/home/username/mysite'  # Ganti 'username' dan 'mysite' sesuai path project Anda di PythonAnywhere
if path not in sys.path:
    sys.path.append(path)

from main import app
from a2wsgi import ASGIMiddleware

# PythonAnywhere menggunakan WSGI (variable harus bernama 'application')
application = ASGIMiddleware(app)
