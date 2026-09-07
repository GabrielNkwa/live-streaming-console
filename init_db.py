<<<<<<< HEAD
from app import app, db, User, Camera
import os

def init_database():
    with app.app_context():
        db.create_all()
        
        admin_username = os.environ.get('APP_ADMIN_USER', 'admin')
        admin_password = os.environ.get('APP_ADMIN_PASS', 'admin')
        
=======
from sqlalchemy import inspect, text
from app import app, db, User, Camera
import os

def migrate_camera_owner_id():
    """Add Camera.owner_id to a pre-existing database that predates
    per-user camera ownership, and backfill existing rows to the admin
    account so no data is silently dropped."""
    inspector = inspect(db.engine)
    if 'camera' not in inspector.get_table_names():
        return
    columns = {col['name'] for col in inspector.get_columns('camera')}
    if 'owner_id' in columns:
        return

    print("Migrating existing 'camera' table: adding owner_id column...")
    admin_username = os.environ.get('APP_ADMIN_USER', 'admin')
    with db.engine.begin() as conn:
        conn.execute(text('ALTER TABLE camera ADD COLUMN owner_id INTEGER'))
        admin_row = conn.execute(
            text('SELECT id FROM user WHERE username = :username'),
            {'username': admin_username}
        ).first()
        if admin_row:
            conn.execute(
                text('UPDATE camera SET owner_id = :owner_id WHERE owner_id IS NULL'),
                {'owner_id': admin_row[0]}
            )
    print("Migration complete: existing cameras assigned to admin account")

def init_database():
    with app.app_context():
        migrate_camera_owner_id()
        db.create_all()

        admin_username = os.environ.get('APP_ADMIN_USER', 'admin')
        admin_password = os.environ.get('APP_ADMIN_PASS', 'admin')

>>>>>>> bd986e8dfa8018f60aaf8135479d221ff2985201
        if not User.query.filter_by(username=admin_username).first():
            admin_user = User(username=admin_username, role='admin')
            admin_user.set_password(admin_password)
            db.session.add(admin_user)
            db.session.commit()
            print(f"Default admin user created: {admin_username}")
<<<<<<< HEAD
        
        if not Camera.query.first():
            default_cam = Camera(name='Default Camera', url='webcam:0')
            db.session.add(default_cam)
            db.session.commit()
            print("Default camera created")
        
=======

        admin_user = User.query.filter_by(username=admin_username).first()
        if not Camera.query.first():
            default_cam = Camera(owner_id=admin_user.id, name='Default Camera', url='webcam:0')
            db.session.add(default_cam)
            db.session.commit()
            print("Default camera created")

>>>>>>> bd986e8dfa8018f60aaf8135479d221ff2985201
        print("Database initialized successfully!")

if __name__ == '__main__':
    init_database()
