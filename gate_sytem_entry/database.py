# import os

# import psycopg2
# from dotenv import load_dotenv


# load_dotenv()

# DATABASE_URL = os.getenv("DATABASE_URL")


# def get_db_connection():
#     if not DATABASE_URL:
#         raise RuntimeError("DATABASE_URL is not set")

#     return psycopg2.connect(DATABASE_URL)


# def _to_pgvector_literal(values):
#     if values is None:
#         return None

#     if hasattr(values, "tolist"):
#         values = values.tolist()

#     return "[" + ",".join(str(float(value)) for value in values) + "]"



# def save_vehicle_entry_record(
#     employee_id,
#     visit_id,
#     vehicle_embedding,
#     vehicle_class,
#     plate_number=None,
#     camera_id=None,
# ):
#     conn = get_db_connection()
#     cursor = conn.cursor()

#     cursor.execute(
#         """
#         INSERT INTO entry_records (
#             visit_id,
#             employee_id,
#             vehicle_embedding,
#             vehicle_class,
#             plate_number,
#             camera_id
#         )
#         VALUES (%s, %s, %s::vector, %s, %s, %s)
#         RETURNING id
#         """,
#         (
#             visit_id,
#             employee_id,
#             _to_pgvector_literal(vehicle_embedding),
#             vehicle_class,
#             plate_number,
#             camera_id,
#         )
#     )

#     entry_id = cursor.fetchone()[0]

#     conn.commit()
#     cursor.close()
#     conn.close()

#     return entry_id




import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")

    return psycopg2.connect(DATABASE_URL)


def _to_pgvector_literal(values):
    if values is None:
        return None

    if hasattr(values, "tolist"):
        values = values.tolist()

    return "[" + ",".join(str(float(value)) for value in values) + "]"


def get_active_visit_id(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT visit_id
        FROM employee_visits
        WHERE employee_id = %s
          AND status = 'ACTIVE'
        ORDER BY entry_time DESC
        LIMIT 1
        """,
        (employee_id,)
    )

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row is None:
        return None

    return row[0]


def save_vehicle_entry_record(
    employee_id,
    visit_id,
    vehicle_embedding,
    vehicle_class,
    plate_number=None,
    camera_id=None,
):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO entry_records (
            visit_id,
            employee_id,
            vehicle_embedding,
            vehicle_class,
            plate_number,
            camera_id
        )
        VALUES (%s, %s, %s::vector, %s, %s, %s)
        RETURNING id
        """,
        (
            visit_id,
            employee_id,
            _to_pgvector_literal(vehicle_embedding),
            vehicle_class,
            plate_number,
            camera_id,
        )
    )

    entry_id = cursor.fetchone()[0]

    conn.commit()
    cursor.close()
    conn.close()

    return entry_id


def save_employee_embedding(
    employee_id,
    mean_embedding,
    all_embeddings,
    embeddings_count
):
    conn = get_db_connection()

    cursor = conn.cursor()

    mean_embedding_list = _to_pgvector_literal(mean_embedding)

    cursor.execute(
        """
        INSERT INTO employees (
            employee_id,
            mean_embedding,
            embeddings_count
        )
        VALUES (%s, %s::vector, %s)
        ON CONFLICT (employee_id)
        DO UPDATE SET
            mean_embedding = EXCLUDED.mean_embedding,
            embeddings_count = EXCLUDED.embeddings_count
        """,
        (
            employee_id,
            mean_embedding_list,
            embeddings_count
        )
    )

    for embedding in all_embeddings:
        embedding_list = embedding.tolist()

        cursor.execute(
            """
            INSERT INTO employee_embeddings (
                employee_id,
                embedding
            )
            VALUES (%s, %s)
            """,
            (
                employee_id,
                embedding_list
            )
        )

    conn.commit()

    cursor.close()
    conn.close()
    
    
def get_all_employee_embeddings():
    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            employee_id,
            embedding
        FROM employee_embeddings
        """
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows



def update_entry_plate(entry_id, plate_number):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE entry_records
        SET plate_number = %s
        WHERE id = %s
        """,
        (plate_number, entry_id)
    )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Successfully updated entry {entry_id} with plate: {plate_number}")



import os
import uuid
import logging
import psycopg2
from psycopg2 import pool
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
logger = logging.getLogger("vehicle_service.database")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is missing from your configuration.")

# Threaded connection pooling to protect serverless performance limits
try:
    db_pool = pool.ThreadedConnectionPool(1, 10, dsn=DATABASE_URL)
    logger.info("Connection pool successfully hooked up to Neon.")
except Exception as e:
    logger.critical(f"Database connection pool initialization aborted: {e}")
    raise


def _to_pgvector_literal(values):
    """Parses standard lists or numpy tensor objects into pgvector compatible string layouts."""
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def get_active_visit_id(employee_id: str) -> str:
    """Checks for a live active tracking session inside the platform database context."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT visit_id FROM employee_visits
                WHERE employee_id = %s AND status = 'ACTIVE'
                ORDER BY entry_time DESC LIMIT 1
                """,
                (employee_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"Error checking active tracking session for {employee_id}: {e}")
        return None
    finally:
        db_pool.putconn(conn)


def create_active_visit_session(employee_id: str, shift_hours: int = 9) -> str:
    """Generates an explicit session record tracking framework using strict unique UUID mapping strings."""
    conn = db_pool.getconn()
    try:
        new_visit_id = str(uuid.uuid4())
        entry_time = datetime.now()
        expected_expiry = entry_time + timedelta(hours=shift_hours)
        
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO employee_visits (visit_id, employee_id, entry_time, expected_expiry, status)
                VALUES (%s, %s, %s, %s, 'ACTIVE')
                """,
                (new_visit_id, employee_id, entry_time, expected_expiry)
            )
            conn.commit()
            logger.info(f"Initialized new tracking shift UUID session context: {new_visit_id}")
            return new_visit_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed session injection process for employee {employee_id}: {e}")
        raise e
    finally:
        db_pool.putconn(conn)


def save_vehicle_entry_record(
    employee_id: str, visit_id: str, vehicle_embedding, vehicle_class: str, plate_number: str = None, camera_id: str = None
) -> int:
    """Commits vector embeddings and classifications directly down to Neon storage arrays."""
    conn = db_pool.getconn()
    try:
        vector_str = _to_pgvector_literal(vehicle_embedding)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO entry_records (visit_id, employee_id, vehicle_embedding, vehicle_class, plate_number, camera_id)
                VALUES (%s, %s, %s::vector, %s, %s, %s)
                RETURNING id
                """,
                (visit_id, employee_id, vector_str, vehicle_class, plate_number, camera_id)
            )
            entry_id = cursor.fetchone()[0]
            conn.commit()
            return entry_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Data entry sync rejected by storage nodes: {e}")
        raise e
    finally:
        db_pool.putconn(conn)
        
        
        
def log_plate_recognition(entry_id: int, plate_number: str):
    """Updates existing entry records with license plate information post-ALPR processing."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE entry_records 
                SET plate_number = %s
                WHERE id = %s
                """,
                (plate_number, entry_id)
            )
            conn.commit()
            logger.info(f"Successfully updated entry {entry_id} with plate: {plate_number}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update entry {entry_id} with plate {plate_number}: {e}")
        raise e
    finally:
        db_pool.putconn(conn)
        
        
        
def complete_visit(visit_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE employee_visits
        SET status = 'COMPLETED'
        WHERE visit_id = %s
        """,
        (visit_id,)
    )
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"Visit completed: {visit_id}")
    


def create_exit_record(
    visit_id,
    employee_id,
    vehicle_embedding,
    vehicle_class,
    plate_number,
    face_verified,
    vehicle_verified,
    plate_verified,
    gate_opened,
    camera_id
):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO exit_records(
            visit_id,
            employee_id,
            vehicle_embedding,
            vehicle_class,
            plate_number,
            face_verified,
            vehicle_verified,
            plate_verified,
            gate_opened,
            camera_id
        )
        VALUES (
            %s,%s,%s::vector,%s,%s,%s,%s,%s,%s,%s
        )
        RETURNING id
        """,
        (
            visit_id,
            employee_id,
            _to_pgvector_literal(vehicle_embedding),
            vehicle_class,
            plate_number,
            face_verified,
            vehicle_verified,
            plate_verified,
            gate_opened,
            camera_id
        )
    )

    exit_id = cursor.fetchone()[0]

    conn.commit()
    cursor.close()
    conn.close()

    return exit_id


def get_active_employee_embeddings():
    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            ee.employee_id,
            ev.visit_id,
            ee.embedding
        FROM employee_embeddings ee
        INNER JOIN employee_visits ev
            ON ee.employee_id = ev.employee_id
        WHERE ev.status = 'ACTIVE'
        """
    )
    
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return rows



def get_active_vehicle_records():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            er.employee_id,
            er.visit_id,
            er.vehicle_embedding,
            er.plate_number
        FROM entry_records er
        INNER JOIN employee_visits ev
            ON er.visit_id = ev.visit_id
        WHERE ev.status = 'ACTIVE'
        """
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows
    
    
    
def complete_visit(visit_id):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE employee_visits
            SET
                status = 'COMPLETED',
                exit_time = NOW()
            WHERE visit_id = %s AND status = 'ACTIVE'
            """,
            (visit_id,)
        )
        
        # Check if any row was actually updated
        if cursor.rowcount == 0:
            print(f"Warning: No ACTIVE visit found for {visit_id}")
            
        conn.commit()
        cursor.close()
    except Exception as e:
        if conn: conn.rollback()
        print(f"Database error during visit completion: {e}")
        raise # Rethrow so the caller knows the operation failed
    finally:
        if conn: conn.close()
        
        
def save_exit_record(visit_id, employee_id, vehicle_embedding, vehicle_class, plate_number, 
                     face_verified, vehicle_verified, plate_verified, gate_opened, camera_id):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO exit_records (
                visit_id, 
                employee_id, 
                vehicle_embedding, 
                vehicle_class, 
                plate_number, 
                face_verified, 
                vehicle_verified, 
                plate_verified, 
                gate_opened, 
                camera_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Note: Ensure vehicle_embedding is a list or numpy array 
        # that the pgvector driver can interpret
        cursor.execute(insert_query, (
            visit_id,
            employee_id,
            vehicle_embedding.tolist() if hasattr(vehicle_embedding, 'tolist') else vehicle_embedding,
            vehicle_class,
            plate_number,
            face_verified,
            vehicle_verified,
            plate_verified,
            gate_opened,
            camera_id
        ))

        conn.commit()
        cursor.close()
        print(f"Exit record saved successfully for visit: {visit_id}")

    except Exception as e:
        if conn: conn.rollback()
        print(f"Error saving exit record to database: {e}")
        raise
    finally:
        if conn: conn.close()