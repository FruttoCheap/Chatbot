import psycopg2
from config import load_config


def create_tables():
    """ Create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE expenses (
            expense_id SERIAL PRIMARY KEY,
            price REAL NOT NULL,
            description VARCHAR(255) NOT NULL,
            category VARCHAR(255) NOT NULL,
            created_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,)
    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                for command in commands:
                    print(command)
                    cur.execute(command)
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


if __name__ == '__main__':
    create_tables()
