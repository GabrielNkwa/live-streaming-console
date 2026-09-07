from app import app, db, User, Camera
import os

def init_database():
    with app.app_context():
        db.create_all()
        
        admin_username = os.environ.get('APP_ADMIN_USER', 'admin')
        admin_password = os.environ.get('APP_ADMIN_PASS', 'admin')
        
        if not User.query.filter_by(username=admin_username).first():
            admin_user = User(username=admin_username, role='admin')
            admin_user.set_password(admin_password)
            db.session.add(admin_user)
            db.session.commit()
            print(f"Default admin user created: {admin_username}")
        
        if not Camera.query.first():
            default_cam = Camera(name='Default Camera', url='webcam:0')
            db.session.add(default_cam)
            db.session.commit()
            print("Default camera created")
        
        print("Database initialized successfully!")

if __name__ == '__main__':
    init_database()
