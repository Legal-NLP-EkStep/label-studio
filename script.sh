docker-compose -f postgres.yml up -d
export POSTGRE_NAME=postgres
export POSTGRE_USER=postgres
export POSTGRE_PASSWORD=1234
export POSTGRE_PORT=5432
export POSTGRE_HOST=localhost
export DJANGO_DB=default
pip install -e .
python label_studio/manage.py migrate
python label_studio/manage.py runserver