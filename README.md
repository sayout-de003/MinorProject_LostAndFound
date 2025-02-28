# Lost and Found System

## Overview
The **Lost and Found System** is a web-based platform built with Django that allows users to report and track lost and found items. The system provides a seamless interface for users to post lost items, check for found items, and connect with others to retrieve their belongings.

## Features
- **User Authentication**: Users can sign up, log in, and manage their accounts.
- **Report Lost Items**: Users can report lost items with details such as description, location, date, and contact information.
- **Report Found Items**: Users can report found items with images and relevant details.
- **Search and Filter**: Users can search for lost and found items using keywords and filters.
- **Notifications**: Get notified when a matching lost/found item is posted.
- **Messaging System**: Contact the person who found a lost item directly through the platform.
- **Admin Dashboard**: Admins can manage reported items and users.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Django
- PostgreSQL or SQLite (for local development)

### Steps to Set Up
1. **Clone the repository:**
   ```bash
   [git clone https://github.com/sayout-de003/MinorProject_LostAndFound]
   cd lost-and-found
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database:**
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser (for admin access):**
   ```bash
   python manage.py createsuperuser
   ```
   Follow the prompts to create an admin account.

6. **Run the development server:**
   ```bash
   python manage.py runserver
   ```

7. **Access the app in your browser:**
   Open `http://127.0.0.1:8000/` in your browser.

## Configuration

### Environment Variables
Create a `.env` file in the project root and set the following:
```env
SECRET_KEY=your_secret_key
DEBUG=True

EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_HOST_USER=your_email@example.com
EMAIL_HOST_PASSWORD=your_email_password
```

## Deployment
To deploy the application:
- Use **Gunicorn** for running the Django app.
- Set up a **PostgreSQL** database in production.
- Configure **Nginx/Apache** to serve static files.
- Use **Docker** if needed for containerization.

## API Endpoints (if applicable)
- `POST /api/lost/` - Report a lost item
- `POST /api/found/` - Report a found item
- `GET /api/items/` - List lost and found items
- `GET /api/items/<id>/` - Get details of a specific item

## Contributing
1. Fork the repository
2. Create a new feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to branch: `git push origin feature-name`
5. Open a pull request

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

## Contact
For any queries, reach out to   1. **dhanrajmahalonia170204@gmail.com**
                                2.**sanyuktakasliwal@gmail.com**
                                3.**cbse821@gmail.com**
                                3.**desayantan1947@gmail.com**
                             

