import sqlite3


DATABASE_NAME = "predictions.db"


def get_connection():
    """
    Create and return a database connection.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create the predictions table if it doesn't exist.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        disease TEXT NOT NULL,

        confidence REAL NOT NULL,

        image TEXT NOT NULL,

        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)

    conn.commit()
    conn.close()


def save_prediction(disease, confidence, image):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions
    (
        disease,
        confidence,
        image
    )
    VALUES (?, ?, ?)
    """,
    (
        disease,
        confidence,
        image
    ))

    conn.commit()
    conn.close()


def get_all_predictions():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM predictions
    ORDER BY prediction_date DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


def delete_prediction(prediction_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions WHERE id=?",
        (prediction_id,)
    )

    conn.commit()
    conn.close()


def clear_history():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()


if __name__ == "__main__":

    init_db()

    print("Database created successfully.")