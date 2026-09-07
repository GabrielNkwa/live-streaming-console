import os
import tempfile
import atexit

# Point the app at a throwaway sqlite file *before* importing it, since
# app.py reads DATABASE_URL at import time when it builds the SQLAlchemy
# engine. An in-memory sqlite URI would give each connection its own
# empty database, so a real (temp) file is used instead.
_db_fd, _db_path = tempfile.mkstemp(suffix='.db')
os.environ['DATABASE_URL'] = f'sqlite:///{_db_path}'
os.environ.setdefault('FLASK_SECRET_KEY', 'test-secret-key-for-pytest')
os.environ['FLASK_ENV'] = 'development'  # relax Secure cookie flag for the test client


def _cleanup_db_file():
    try:
        os.close(_db_fd)
        os.remove(_db_path)
    except OSError:
        pass


atexit.register(_cleanup_db_file)

import time

import pytest
import app as app_module
from app import app, db, User, Camera, get_camera, release_camera_connection

CSRF = 'test-csrf-token'


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.app_context():
        db.drop_all()
        db.create_all()
        admin = User(username='admin', role='admin')
        admin.set_password('adminpass123')
        alice = User(username='alice', role='user')
        alice.set_password('alicepass123')
        bob = User(username='bob', role='user')
        bob.set_password('bobpass123')
        db.session.add_all([admin, alice, bob])
        db.session.commit()

    with app.test_client() as c:
        yield c

    with app.app_context():
        db.session.remove()
        db.drop_all()


class FakeCapture:
    """Stands in for cv2.VideoCapture so the connection-pooling logic can be
    tested without real camera hardware or network access."""

    def __init__(self):
        self.released = False

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True

    def set(self, *args, **kwargs):
        pass

    def grab(self):
        return True

    def retrieve(self):
        return True, object()


@pytest.fixture(autouse=True)
def _clean_camera_pool():
    """The camera connection pool is module-level global state shared across
    the whole test session - reset it around every test so tests can't leak
    fake connections into each other."""
    app_module.active_cameras.clear()
    app_module.camera_last_used.clear()
    app_module.camera_url_locks.clear()
    yield
    app_module.active_cameras.clear()
    app_module.camera_last_used.clear()
    app_module.camera_url_locks.clear()


def _prime_csrf(test_client):
    with test_client.session_transaction() as sess:
        sess['csrf_token'] = CSRF


def login(test_client, username, password):
    _prime_csrf(test_client)
    return test_client.post(
        '/login',
        data={'username': username, 'password': password, 'csrf_token': CSRF},
        follow_redirects=True,
    )


def api_post(test_client, url, json_body=None):
    return test_client.post(url, json=json_body or {}, headers={'X-CSRF-Token': CSRF})


# --- Authentication / authorization -----------------------------------

def test_dashboard_requires_login(client):
    resp = client.get('/dashboard', follow_redirects=False)
    assert resp.status_code in (302, 401)


def test_login_success_and_failure(client):
    resp = login(client, 'alice', 'wrongpassword')
    assert b'Invalid credentials' in resp.data

    resp = login(client, 'alice', 'alicepass123')
    assert resp.status_code == 200
    assert b'My Cameras' in resp.data


def test_non_admin_cannot_reach_admin_routes(client):
    login(client, 'alice', 'alicepass123')

    resp = client.get('/admin', follow_redirects=True)
    assert b'Admin Dashboard' not in resp.data

    resp = api_post(client, '/api/user/add', {'username': 'eve', 'password': 'evepassword', 'role': 'user'})
    assert resp.status_code == 403

    resp = api_post(client, '/api/admin/camera/remove', {'camera_id': 1})
    assert resp.status_code == 403


# --- Per-user camera ownership ------------------------------------------

def test_client_can_add_and_view_own_camera(client):
    login(client, 'alice', 'alicepass123')

    resp = api_post(client, '/api/camera/add', {'camera_name': 'Front Door', 'camera_url': 'webcam:0'})
    assert resp.status_code == 200
    assert resp.get_json()['success'] is True

    resp = client.get('/dashboard')
    assert b'Front Door' in resp.data


def test_users_cannot_see_or_touch_each_others_cameras(client):
    login(client, 'alice', 'alicepass123')
    api_post(client, '/api/camera/add', {'camera_name': 'Alice Cam', 'camera_url': 'webcam:0'})
    client.post('/logout', data={'csrf_token': CSRF})

    login(client, 'bob', 'bobpass123')

    # Bob's dashboard must not list Alice's camera.
    resp = client.get('/dashboard')
    assert b'Alice Cam' not in resp.data

    # Bob cannot stream, update, or remove Alice's camera by name.
    resp = client.get('/video_feed/Alice%20Cam')
    assert resp.status_code == 404

    resp = api_post(client, '/api/camera/remove', {'camera_name': 'Alice Cam'})
    assert resp.status_code == 404

    resp = api_post(client, '/api/camera/update', {
        'old_name': 'Alice Cam', 'new_name': 'Hijacked', 'new_url': 'webcam:1'
    })
    assert resp.status_code == 404


def test_two_users_can_reuse_the_same_camera_name(client):
    login(client, 'alice', 'alicepass123')
    resp = api_post(client, '/api/camera/add', {'camera_name': 'Cam1', 'camera_url': 'webcam:0'})
    assert resp.status_code == 200
    client.post('/logout', data={'csrf_token': CSRF})

    login(client, 'bob', 'bobpass123')
    resp = api_post(client, '/api/camera/add', {'camera_name': 'Cam1', 'camera_url': 'webcam:1'})
    assert resp.status_code == 200


# --- Admin user management ----------------------------------------------

def test_admin_sees_camera_counts_and_can_edit_users(client):
    login(client, 'alice', 'alicepass123')
    api_post(client, '/api/camera/add', {'camera_name': 'Alice Cam', 'camera_url': 'webcam:0'})
    client.post('/logout', data={'csrf_token': CSRF})

    login(client, 'admin', 'adminpass123')
    resp = client.get('/admin')
    assert b'1 camera' in resp.data

    with app.app_context():
        alice_id = User.query.filter_by(username='alice').first().id

    resp = api_post(client, '/api/user/update', {'user_id': alice_id, 'username': 'alice2'})
    assert resp.status_code == 200
    with app.app_context():
        assert User.query.get(alice_id).username == 'alice2'


def test_cannot_demote_the_last_admin(client):
    login(client, 'admin', 'adminpass123')
    with app.app_context():
        admin_id = User.query.filter_by(username='admin').first().id

    resp = api_post(client, '/api/user/update', {'user_id': admin_id, 'role': 'user'})
    assert resp.status_code == 400
    assert 'last remaining admin' in resp.get_json()['error']


def test_remove_user_rejects_non_numeric_id(client):
    login(client, 'admin', 'adminpass123')
    resp = api_post(client, '/api/user/remove', {'user_id': 'not-a-number'})
    assert resp.status_code == 400
    assert resp.get_json()['success'] is False


# --- Camera connection pooling --------------------------------------------

def test_get_camera_reuses_existing_connection(monkeypatch):
    open_calls = []

    def fake_open_capture(url):
        open_calls.append(url)
        return FakeCapture()

    monkeypatch.setattr(app_module, '_open_capture', fake_open_capture)

    cap1 = get_camera('webcam:0')
    cap2 = get_camera('webcam:0')

    assert cap1 is cap2
    assert open_calls == ['webcam:0']  # only connected once, second call reused the pool


def test_get_camera_returns_none_when_open_fails(monkeypatch):
    monkeypatch.setattr(app_module, '_open_capture', lambda url: None)
    assert get_camera('webcam:0') is None
    assert 'webcam:0' not in app_module.active_cameras


def test_get_camera_evicts_lru_when_at_capacity(monkeypatch):
    monkeypatch.setattr(app_module, '_open_capture', lambda url: FakeCapture())
    monkeypatch.setattr(app_module, 'MAX_ACTIVE_CAMERAS', 2)
    monkeypatch.setattr(app_module, 'CAMERA_EVICTION_GRACE_SECONDS', 0)

    cap_a = get_camera('webcam:0')
    cap_b = get_camera('webcam:1')
    assert len(app_module.active_cameras) == 2

    # Backdate A so it's the clear LRU candidate.
    app_module.camera_last_used['webcam:0'] = time.monotonic() - 999

    cap_c = get_camera('webcam:2')

    assert 'webcam:0' not in app_module.active_cameras  # evicted
    assert cap_a.released is True
    assert 'webcam:1' in app_module.active_cameras
    assert 'webcam:2' in app_module.active_cameras
    assert len(app_module.active_cameras) == 2


def test_get_camera_refuses_new_connection_when_all_recently_used(monkeypatch):
    monkeypatch.setattr(app_module, '_open_capture', lambda url: FakeCapture())
    monkeypatch.setattr(app_module, 'MAX_ACTIVE_CAMERAS', 1)
    monkeypatch.setattr(app_module, 'CAMERA_EVICTION_GRACE_SECONDS', 999)

    get_camera('webcam:0')
    result = get_camera('webcam:1')  # webcam:0 was just used, can't evict yet

    assert result is None
    assert 'webcam:1' not in app_module.active_cameras


def test_release_camera_connection_keeps_connection_if_still_referenced(client):
    login(client, 'alice', 'alicepass123')
    api_post(client, '/api/camera/add', {'camera_name': 'Cam1', 'camera_url': 'webcam:0'})
    client.post('/logout', data={'csrf_token': CSRF})
    login(client, 'bob', 'bobpass123')
    api_post(client, '/api/camera/add', {'camera_name': 'Cam1', 'camera_url': 'webcam:0'})

    fake_cap = FakeCapture()
    with app.app_context():
        app_module.active_cameras['webcam:0'] = fake_cap
        app_module.camera_last_used['webcam:0'] = time.monotonic()
        # Both alice's and bob's cameras point at webcam:0; removing alice's
        # copy must not kill the connection bob is still using.
        release_camera_connection('webcam:0')
        assert fake_cap.released is False
        assert 'webcam:0' in app_module.active_cameras

        # Removing bob's row still leaves alice's Cam1 pointing at webcam:0,
        # so the connection must survive.
        bob_user = User.query.filter_by(username='bob').first()
        bob_cam = Camera.query.filter_by(owner_id=bob_user.id, name='Cam1').first()
        db.session.delete(bob_cam)
        db.session.commit()
        release_camera_connection('webcam:0')
        assert fake_cap.released is False
        assert 'webcam:0' in app_module.active_cameras

        # Now remove alice's too - the last remaining reference - and only
        # then should the pooled connection actually be released.
        alice_user = User.query.filter_by(username='alice').first()
        alice_cam = Camera.query.filter_by(owner_id=alice_user.id, name='Cam1').first()
        db.session.delete(alice_cam)
        db.session.commit()
        release_camera_connection('webcam:0')
        assert fake_cap.released is True
        assert 'webcam:0' not in app_module.active_cameras


def test_logout_does_not_kill_other_users_active_cameras(client):
    login(client, 'alice', 'alicepass123')
    fake_cap = FakeCapture()
    app_module.active_cameras['webcam:0'] = fake_cap
    app_module.camera_last_used['webcam:0'] = time.monotonic()

    client.post('/logout', data={'csrf_token': CSRF})

    assert fake_cap.released is False
    assert 'webcam:0' in app_module.active_cameras
