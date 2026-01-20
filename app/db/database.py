import psycopg2
import os

DB_CONFIG = {
    'host': 'localhost',
    'dbname': 'ml_monitoring',
    'user': 'postgres',
    'password': 'anand',
    'port': 5432
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)