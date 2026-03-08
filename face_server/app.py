#!/usr/bin/env python3
"""
app.py
Server sederhana untuk menyimpan / melayani foto siswa.

Pola penyimpanan:
    {DATA_DIR}/{id_siswa}/{file}

Pastikan di .env:
    DATA_DIR=/home/pi/capstone1/face_database
    PI_API_KEY=...
    PI_BASE_URL=http://your.host:5050
    MAX_FILE_SIZE_KB=5120

Gunakan gunicorn/uwsgi di production; `app.run()` hanya untuk testing lokal.
"""
import os
import uuid
import mimetypes
from pathlib import Path
from flask import Flask, request, jsonify, send_file, abort, current_app
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS

# --- Load environment ---
load_dotenv()

# ENV dan default
env_data_dir = os.getenv('DATA_DIR', './data')  # default lokal untuk testing
project_root = Path(__file__).parent.resolve()
fallback_local = project_root / 'data'

# Buat folder DATA_DIR atau fallback
try:
    DATA_DIR = Path(env_data_dir).expanduser().resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except (PermissionError, FileNotFoundError) as e:
    current_app = None
    try:
        # Try to log to console even if Flask app not created yet
        print(f"[warning] cannot create DATA_DIR at '{env_data_dir}' ({e}). Falling back to {fallback_local}")
    except Exception:
        pass
    try:
        fallback_local.mkdir(parents=True, exist_ok=True)
    except Exception as e2:
        raise RuntimeError(f"Failed to create fallback data dir '{fallback_local}': {e2}") from e2
    DATA_DIR = fallback_local
except Exception as e:
    raise

API_KEY = os.getenv('PI_API_KEY', 'change_me')
# BASE_URL digunakan untuk membentuk URL direct yang dikembalikan di JSON listing
BASE_URL = os.getenv('PI_BASE_URL', 'http://127.0.0.1:5050').rstrip('/')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE_KB', '5120')) * 1024
ALLOWED_EXT = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ProxyFix jika jadi di belakang reverse-proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# --- CORS SETUP (single source of truth from env) ---
cors_env = os.getenv('CORS_ORIGINS')  # comma-separated or '*'
if cors_env:
    origins = [o.strip() for o in cors_env.split(',')]
    if len(origins) == 1 and origins[0] == '*':
        CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
    else:
        CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=False)
else:
    # sensible default for local testing
    CORS(app, resources={r"/*": {"origins": [BASE_URL]}}, supports_credentials=False)
# --- END CORS SETUP ---

def check_api_key(req):
    key = req.headers.get('X-API-KEY') or req.args.get('api_key')
    return key and key == API_KEY

def safe_join(base: Path, *paths):
    """
    Join paths under `base` safely preventing directory traversal.

    - Strips leading slashes from path components to avoid absolute path hijack.
    - Resolves both base and target and ensures target is inside base.
    """
    base_resolved = base.resolve()
    # sanitize components (remove leading slashes to avoid absolute join behavior)
    parts = [str(p).lstrip('/') for p in paths if p is not None]
    target = base_resolved.joinpath(*parts).resolve()
    # allow if target == base or target is inside base
    if target == base_resolved or base_resolved in target.parents:
        return target
    raise ValueError('Invalid path')

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'ok': False, 'message': 'File terlalu besar'}), 413

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'msg': 'pi server ok', 'DATA_DIR': str(DATA_DIR)})

@app.route('/api/debug-info', methods=['GET'])
def debug_info():
    if not check_api_key(request):
        return jsonify({'ok': False, 'message': 'Unauthorized'}), 401
    return jsonify({
        'ok': True,
        'DATA_DIR': str(DATA_DIR),
        'env_DATA_DIR': os.getenv('DATA_DIR'),
        'BASE_URL': BASE_URL,
        'api_key_set': bool(os.getenv('PI_API_KEY')),
    })

@app.route('/api/check-file', methods=['GET'])
def check_file():
    if not check_api_key(request):
        return jsonify({'ok': False, 'message': 'Unauthorized'}), 401
    path = request.args.get('path') or ''
    if path.startswith('/'):
        path = path.lstrip('/')
    try:
        target = safe_join(DATA_DIR, path)
    except ValueError:
        return jsonify({'ok': False, 'message': 'Invalid path'}), 400
    exists = target.exists() and target.is_file()
    return jsonify({'ok': True, 'path': path, 'resolved': str(target), 'exists': exists})

@app.route('/api/list-photos/<siswa_id>', methods=['GET'])
def list_photos(siswa_id):
    """Return JSON listing of files for student's folder.

    Folder layout used now: {DATA_DIR}/{siswa_id}/
    """
    if not check_api_key(request):
        return jsonify({'ok': False, 'message': 'Unauthorized'}), 401

    try:
        folder = safe_join(DATA_DIR, str(siswa_id))
    except ValueError:
        return jsonify({'ok': False, 'message': 'Invalid path'}), 400

    if not folder.exists() or not folder.is_dir():
        return jsonify({'ok': False, 'message': 'Folder not found', 'files': []}), 404

    items = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        try:
            stat = p.stat()
            rel = p.relative_to(DATA_DIR).as_posix()  # e.g. "12345/photo.jpg"
            url = f"{BASE_URL}/{rel}"
            items.append({
                'name': p.name,
                'path': rel,
                'size': stat.st_size,
                'url': url
            })
        except Exception as e:
            current_app.logger.debug("list_photos skip error: %s", e)

    return jsonify(items), 200

@app.route('/api/upload-photo', methods=['POST'])
def upload_photo():
    if not check_api_key(request):
        return jsonify({'ok': False, 'message': 'Unauthorized'}), 401

    if 'file' not in request.files:
        return jsonify({'ok': False, 'message': 'No file supplied'}), 400

    file = request.files['file']
    siswa_id = request.form.get('siswa_id') or request.args.get('siswa_id') or 'unknown'
    try:
        siswa_dir = safe_join(DATA_DIR, str(siswa_id))
    except ValueError:
        return jsonify({'ok': False, 'message': 'Invalid siswa_id'}, 400)
    siswa_dir.mkdir(parents=True, exist_ok=True)

    orig_name = secure_filename(file.filename or 'upload.jpg')
    if '.' in orig_name:
        name_part, ext = orig_name.rsplit('.', 1)
        ext = ext.lower()
    else:
        name_part, ext = orig_name, 'jpg'

    if ext not in ALLOWED_EXT:
        return jsonify({'ok': False, 'message': 'Tipe file tidak diperbolehkan'}), 400

    new_name = f"{uuid.uuid4().hex}_{name_part}.{ext}"
    dest = siswa_dir / new_name

    try:
        # save file
        file.save(str(dest))

        # quick validation: is it an image?
        try:
            with Image.open(str(dest)) as im:
                im.verify()
        except (UnidentifiedImageError, Exception):
            dest.unlink(missing_ok=True)
            return jsonify({'ok': False, 'message': 'File bukan gambar valid'}), 400

        # read format reliably
        try:
            with Image.open(str(dest)) as im2:
                fmt = (im2.format or '').lower()
        except Exception:
            dest.unlink(missing_ok=True)
            return jsonify({'ok': False, 'message': 'Gagal membaca format gambar'}), 400

        if fmt not in ('jpeg', 'png'):
            dest.unlink(missing_ok=True)
            return jsonify({'ok': False, 'message': 'Gambar tidak dikenali (bukan JPEG/PNG)'}), 400

        # Build returned paths relative to DATA_DIR (leading slash for convenience)
        rel_path = f"/{siswa_id}/{new_name}"
        url_direct = f"{BASE_URL}/{siswa_id}/{new_name}"
        abs_path = str(dest.resolve())
        current_app.logger.info("upload_photo saved %s", abs_path)

        return jsonify({
            'ok': True,
            'path': rel_path,
            'url': url_direct,
            'saved_as': abs_path
        }), 200

    except Exception as e:
        current_app.logger.exception("upload error")
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({'ok': False, 'message': 'Gagal menyimpan file'}), 500

@app.route('/api/delete-photo', methods=['POST'])
def delete_photo():
    if not check_api_key(request):
        return jsonify({'ok': False, 'message': 'Unauthorized'}), 401

    data = request.get_json(silent=True) or request.form
    path = data.get('path') if isinstance(data, dict) else None
    if not path:
        return jsonify({'ok': False, 'message': 'Missing path'}), 400

    try:
        if path.startswith('/'):
            path = path.lstrip('/')
        target = safe_join(DATA_DIR, path)
        if not target.exists():
            return jsonify({'ok': False, 'message': 'File tidak ditemukan'}), 404
        target.unlink()
        parent = target.parent
        # Remove parent if empty and not equal to DATA_DIR
        try:
            if parent != DATA_DIR and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass
        return jsonify({'ok': True, 'message': 'Deleted'}), 200
    except ValueError:
        return jsonify({'ok': False, 'message': 'Invalid path'}), 400
    except Exception as e:
        current_app.logger.exception("delete error")
        return jsonify({'ok': False, 'message': 'Gagal menghapus file'}), 500

@app.route('/api/make-folder', methods=['POST'])
def make_folder():
    if not check_api_key(request):
        return jsonify({'ok': False, 'message': 'Unauthorized'}), 401

    data = request.get_json(silent=True) or request.form
    path = data.get('path') if isinstance(data, dict) else None
    if not path:
        return jsonify({'ok': False, 'message': 'Missing path parameter'}), 400

    path = path.lstrip('/')
    try:
        target = safe_join(DATA_DIR, path)
    except ValueError:
        return jsonify({'ok': False, 'message': 'Invalid path'}), 400

    try:
        target.mkdir(parents=True, exist_ok=True)
        return jsonify({'ok': True, 'path': path, 'message': 'Folder created (or already exists)'}), 200
    except Exception as e:
        current_app.logger.exception("make_folder error")
        return jsonify({'ok': False, 'message': 'Failed to create folder', 'detail': str(e)}), 500

# Serve files directly under DATA_DIR via HTTP GET /<path>
@app.route('/<path:filepath>', methods=['GET'])
def serve_direct(filepath):
    # require api key for direct file access
    if not check_api_key(request):
        return jsonify({'ok': False, 'message': 'Unauthorized'}), 401

    # validate input early
    if filepath.startswith('/'):
        filepath = filepath.lstrip('/')
    if '..' in filepath or '\\' in filepath:
        abort(400)

    try:
        target = safe_join(DATA_DIR, filepath)
    except ValueError:
        abort(400)

    if not target.exists() or not target.is_file():
        current_app.logger.debug("serve_direct: file not found: %s", str(target))
        abort(404)

    # Guess mime type
    ctype, _ = mimetypes.guess_type(str(target))
    if not ctype:
        ctype = 'application/octet-stream'

    try:
        return send_file(str(target), mimetype=ctype)
    except HTTPException:
        raise
    except Exception as e:
        current_app.logger.exception("serve_direct error")
        abort(500)

if __name__ == '__main__':
    # NOTE: for quick local testing only. Use gunicorn in production runs.
    host = os.getenv('PI_HOST', '127.0.0.1')
    port = int(os.getenv('PI_PORT', '5050'))
    app.run(host=host, port=port, debug=False)
