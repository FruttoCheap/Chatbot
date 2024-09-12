import psycopg2

from google.db.config import load_config


class Expense:
    def __init__(self, price, description, category):
        self.price = price
        self.description = description
        self.category = category


def insert_vendor(expense: Expense):
    """ Insert a new vendor into the vendors table """

    sql = """INSERT INTO expenses(price, description, category)
             VALUES(%s, %s, %s) RETURNING expense_id;"""

    expense_id = None
    config = load_config()

    try:
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (expense.price, expense.description, expense.category))

                rows = cur.fetchone()
                if rows:
                    expense_id = rows[0]

                conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        return expense_id


if __name__ == '__main__':
    expense = Expense(100, "Food", "Groceries")
    insert_vendor(expense)
