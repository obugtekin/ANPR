import sqlite3
from datetime import datetime

from PyQt5.QtCore import QObject, pyqtSignal


class DatabaseManager(QObject):
    save_plate_signal = pyqtSignal(str)
    cars_inside = True  # Flag to indicate if cars are inside or outside

    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect('plates.db')
        self.cursor = self.conn.cursor()
        self.create_tables()  # Create all necessary tables

        # Connect the signal to save_plate_to_database slot
        self.save_plate_signal.connect(self.save_plate_to_database)

    def create_tables(self):
        # Create tables if they don't exist
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS CARSIN (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                plate_text TEXT,
                                timestamp TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS CARSINSIDE (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                plate_text TEXT,
                                timestamp TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS CARSOUT (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                plate_text TEXT,
                                timestamp TEXT)''')

    def clean_plate_text(self, plate_text):
        # Remove all characters except numbers and letters
        return ''.join(c for c in plate_text if c.isalnum())

    def save_plate_to_database(self, plate_text):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cleaned_plate_text = self.clean_plate_text(plate_text)

        table_name = "CARSIN" if self.cars_inside else "CARSOUT"  # Determine which table to save based on cars_inside flag
        self.cursor.execute(f"INSERT INTO {table_name} (plate_text, timestamp) VALUES (?, ?)",
                            (cleaned_plate_text, current_datetime))
        self.conn.commit()

        # If cars are inside, also save to CARSINSIDE table
        if self.cars_inside:
            self.cursor.execute("INSERT INTO CARSINSIDE (plate_text, timestamp) VALUES (?, ?)",
                                (cleaned_plate_text, current_datetime))
            self.conn.commit()
        else:
            # If cars are outside, delete the plate from CARSINSIDE
            self.delete_plate_from_inside(cleaned_plate_text)

    def delete_plate_from_inside(self, plate_text):
        # Delete the plate from CARSINSIDE table
        self.cursor.execute("DELETE FROM CARSINSIDE WHERE plate_text=?", (plate_text,))
        self.conn.commit()

    def get_latest_plates(self, table_name, limit=50):
        try:
            self.cursor.execute(f"SELECT plate_text, timestamp FROM {table_name} ORDER BY timestamp DESC LIMIT {limit}")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error fetching plates from {table_name}:", e)
            return []
