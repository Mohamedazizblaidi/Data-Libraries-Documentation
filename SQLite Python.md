# SQLite Python Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Getting Started](#getting-started)
5. [Connection Management](#connection-management)
6. [Creating Tables](#creating-tables)
7. [CRUD Operations](#crud-operations)
8. [Query Operations](#query-operations)
9. [Transaction Management](#transaction-management)
10. [Error Handling](#error-handling)
11. [Advanced Features](#advanced-features)
12. [Performance Optimization](#performance-optimization)
13. [Security Considerations](#security-considerations)
14. [Best Practices](#best-practices)
15. [Examples](#examples)
16. [API Reference](#api-reference)

## Introduction

SQLite is a lightweight, serverless, self-contained SQL database engine. Python's `sqlite3` module provides a comprehensive interface to SQLite databases, allowing you to create, read, update, and delete data using standard SQL commands.

### Key Features

- **Serverless**: No separate server process required
- **Self-contained**: Entire database stored in a single file
- **Zero-configuration**: No setup or administration needed
- **Cross-platform**: Works on all major operating systems
- **ACID compliant**: Supports atomic, consistent, isolated, and durable transactions

## Installation

The `sqlite3` module is included in Python's standard library, so no additional installation is required.

```python
import sqlite3
```

## Basic Concepts

### Database Connection
A connection object represents a database connection and provides methods for executing SQL commands.

### Cursor
A cursor object is used to execute SQL commands and fetch results from queries.

### Transactions
SQLite supports transactions that allow you to group multiple operations together.

## Getting Started

### Creating a Database Connection

```python
import sqlite3

# Connect to a database file (creates it if it doesn't exist)
conn = sqlite3.connect('example.db')

# Connect to an in-memory database
conn = sqlite3.connect(':memory:')

# Always close the connection when done
conn.close()
```

### Using Context Managers

```python
import sqlite3

# Recommended approach using context manager
with sqlite3.connect('example.db') as conn:
    # Database operations here
    cursor = conn.cursor()
    # Connection automatically closed when exiting the block
```

## Connection Management

### Connection Parameters

```python
import sqlite3

# Basic connection
conn = sqlite3.connect('database.db')

# Connection with timeout
conn = sqlite3.connect('database.db', timeout=20.0)

# Connection with custom isolation level
conn = sqlite3.connect('database.db', isolation_level='DEFERRED')

# Read-only connection
conn = sqlite3.connect('file:database.db?mode=ro', uri=True)
```

### Connection Properties

```python
# Set row factory for dictionary-like access
conn.row_factory = sqlite3.Row

# Enable foreign key constraints
conn.execute('PRAGMA foreign_keys = ON')

# Set text factory
conn.text_factory = str
```

## Creating Tables

### Basic Table Creation

```python
import sqlite3

with sqlite3.connect('example.db') as conn:
    cursor = conn.cursor()
    
    # Create a simple table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
```

### Advanced Table Creation

```python
# Table with foreign key constraints
cursor.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT,
        user_id INTEGER,
        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')

# Table with indexes
cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON users (username)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_posts ON posts (user_id)')
```

## CRUD Operations

### Create (Insert) Operations

```python
import sqlite3

with sqlite3.connect('example.db') as conn:
    cursor = conn.cursor()
    
    # Insert single record
    cursor.execute(
        "INSERT INTO users (username, email) VALUES (?, ?)",
        ('john_doe', 'john@example.com')
    )
    
    # Insert with named parameters
    cursor.execute(
        "INSERT INTO users (username, email) VALUES (:username, :email)",
        {'username': 'jane_doe', 'email': 'jane@example.com'}
    )
    
    # Insert multiple records
    users_data = [
        ('alice', 'alice@example.com'),
        ('bob', 'bob@example.com'),
        ('charlie', 'charlie@example.com')
    ]
    cursor.executemany(
        "INSERT INTO users (username, email) VALUES (?, ?)",
        users_data
    )
    
    # Get the ID of the last inserted row
    last_id = cursor.lastrowid
    print(f"Last inserted ID: {last_id}")
    
    conn.commit()
```

### Read (Select) Operations

```python
import sqlite3

with sqlite3.connect('example.db') as conn:
    conn.row_factory = sqlite3.Row  # Enable dictionary-like access
    cursor = conn.cursor()
    
    # Select all records
    cursor.execute("SELECT * FROM users")
    all_users = cursor.fetchall()
    
    # Select with conditions
    cursor.execute("SELECT * FROM users WHERE username = ?", ('john_doe',))
    user = cursor.fetchone()
    
    # Select with multiple conditions
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND email LIKE ?",
        ('john_doe', '%@example.com')
    )
    
    # Select with LIMIT
    cursor.execute("SELECT * FROM users LIMIT 5")
    limited_users = cursor.fetchall()
    
    # Iterate through results
    cursor.execute("SELECT * FROM users")
    for row in cursor:
        print(f"User: {row['username']}, Email: {row['email']}")
```

### Update Operations

```python
import sqlite3

with sqlite3.connect('example.db') as conn:
    cursor = conn.cursor()
    
    # Update single record
    cursor.execute(
        "UPDATE users SET email = ? WHERE username = ?",
        ('newemail@example.com', 'john_doe')
    )
    
    # Update multiple records
    cursor.execute(
        "UPDATE users SET email = ? WHERE email LIKE ?",
        ('updated@example.com', '%@oldomain.com')
    )
    
    # Check how many rows were affected
    rows_affected = cursor.rowcount
    print(f"Rows updated: {rows_affected}")
    
    conn.commit()
```

### Delete Operations

```python
import sqlite3

with sqlite3.connect('example.db') as conn:
    cursor = conn.cursor()
    
    # Delete specific record
    cursor.execute("DELETE FROM users WHERE username = ?", ('john_doe',))
    
    # Delete with multiple conditions
    cursor.execute(
        "DELETE FROM users WHERE username = ? AND email = ?",
        ('jane_doe', 'jane@example.com')
    )
    
    # Delete all records (be careful!)
    # cursor.execute("DELETE FROM users")
    
    rows_deleted = cursor.rowcount
    print(f"Rows deleted: {rows_deleted}")
    
    conn.commit()
```

## Query Operations

### Advanced Queries

```python
import sqlite3

with sqlite3.connect('example.db') as conn:
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # JOIN operations
    cursor.execute('''
        SELECT u.username, p.title, p.created_date
        FROM users u
        JOIN posts p ON u.id = p.user_id
        ORDER BY p.created_date DESC
    ''')
    
    # Aggregate functions
    cursor.execute('''
        SELECT COUNT(*) as total_users,
               AVG(LENGTH(username)) as avg_username_length
        FROM users
    ''')
    result = cursor.fetchone()
    print(f"Total users: {result['total_users']}")
    
    # GROUP BY and HAVING
    cursor.execute('''
        SELECT user_id, COUNT(*) as post_count
        FROM posts
        GROUP BY user_id
        HAVING COUNT(*) > 1
    ''')
    
    # Subqueries
    cursor.execute('''
        SELECT username
        FROM users
        WHERE id IN (
            SELECT DISTINCT user_id
            FROM posts
            WHERE title LIKE '%Python%'
        )
    ''')
```

### Parameterized Queries

```python
# Always use parameterized queries to prevent SQL injection
def get_user_posts(username, limit=10):
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.title, p.content, p.created_date
            FROM posts p
            JOIN users u ON p.user_id = u.id
            WHERE u.username = ?
            ORDER BY p.created_date DESC
            LIMIT ?
        ''', (username, limit))
        
        return cursor.fetchall()
```

## Transaction Management

### Manual Transaction Control

```python
import sqlite3

conn = sqlite3.connect('example.db')

try:
    cursor = conn.cursor()
    
    # Begin transaction (implicit)
    cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)",
                  ('user1', 'user1@example.com'))
    
    cursor.execute("INSERT INTO posts (title, content, user_id) VALUES (?, ?, ?)",
                  ('Post Title', 'Post content', cursor.lastrowid))
    
    # Commit transaction
    conn.commit()
    print("Transaction completed successfully")
    
except sqlite3.Error as e:
    # Rollback on error
    conn.rollback()
    print(f"Transaction failed: {e}")
    
finally:
    conn.close()
```

### Using Context Managers for Transactions

```python
import sqlite3

# Context manager automatically handles commit/rollback
with sqlite3.connect('example.db') as conn:
    cursor = conn.cursor()
    
    # All operations within this block are part of a transaction
    cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)",
                  ('user2', 'user2@example.com'))
    
    cursor.execute("INSERT INTO posts (title, content, user_id) VALUES (?, ?, ?)",
                  ('Another Post', 'More content', cursor.lastrowid))
    
    # Automatically commits on successful exit
    # Automatically rolls back on exception
```

### Savepoints

```python
import sqlite3

with sqlite3.connect('example.db') as conn:
    cursor = conn.cursor()
    
    try:
        # Create savepoint
        cursor.execute("SAVEPOINT sp1")
        
        cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)",
                      ('temp_user', 'temp@example.com'))
        
        # Some condition that might require rollback
        if some_condition:
            cursor.execute("ROLLBACK TO sp1")
        else:
            cursor.execute("RELEASE sp1")
            
    except sqlite3.Error as e:
        cursor.execute("ROLLBACK TO sp1")
        print(f"Error: {e}")
```

## Error Handling

### Common SQLite Exceptions

```python
import sqlite3

try:
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM non_existent_table")
        
except sqlite3.OperationalError as e:
    print(f"Operational error: {e}")
    
except sqlite3.IntegrityError as e:
    print(f"Integrity constraint violated: {e}")
    
except sqlite3.ProgrammingError as e:
    print(f"Programming error: {e}")
    
except sqlite3.DatabaseError as e:
    print(f"Database error: {e}")
    
except sqlite3.Error as e:
    print(f"SQLite error: {e}")
```

### Comprehensive Error Handling

```python
import sqlite3
import logging

def safe_database_operation(query, parameters=None):
    """Execute database operation with comprehensive error handling."""
    try:
        with sqlite3.connect('example.db', timeout=10.0) as conn:
            cursor = conn.cursor()
            
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
                
            return cursor.fetchall()
            
    except sqlite3.IntegrityError as e:
        logging.error(f"Integrity constraint violation: {e}")
        raise
        
    except sqlite3.OperationalError as e:
        logging.error(f"Operational error (possibly locked database): {e}")
        raise
        
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
```

## Advanced Features

### Custom Functions

```python
import sqlite3
import math

# Create custom function
def sqrt_func(x):
    return math.sqrt(x)

with sqlite3.connect(':memory:') as conn:
    # Register custom function
    conn.create_function("SQRT", 1, sqrt_func)
    
    cursor = conn.cursor()
    cursor.execute("SELECT SQRT(16)")
    result = cursor.fetchone()
    print(result[0])  # Output: 4.0
```

### Custom Aggregates

```python
import sqlite3

class GeometricMean:
    def __init__(self):
        self.values = []
    
    def step(self, value):
        self.values.append(value)
    
    def finalize(self):
        if not self.values:
            return None
        product = 1
        for value in self.values:
            product *= value
        return product ** (1.0 / len(self.values))

with sqlite3.connect(':memory:') as conn:
    conn.create_aggregate("GEOMEAN", 1, GeometricMean)
    
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE numbers (value REAL)")
    cursor.executemany("INSERT INTO numbers VALUES (?)", [(2,), (8,), (4,)])
    
    cursor.execute("SELECT GEOMEAN(value) FROM numbers")
    result = cursor.fetchone()
    print(result[0])  # Output: 4.0
```

### Row Factory Customization

```python
import sqlite3

# Custom row factory
def dict_factory(cursor, row):
    """Convert row to dictionary with column names as keys."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

# Using Row factory for named access
with sqlite3.connect('example.db') as conn:
    conn.row_factory = sqlite3.Row  # Built-in Row factory
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users LIMIT 1")
    row = cursor.fetchone()
    
    if row:
        print(row['username'])  # Access by column name
        print(row[0])          # Access by index
        print(dict(row))       # Convert to dictionary
```

### Backup and Restore

```python
import sqlite3

def backup_database(source_db, backup_db):
    """Backup database to another file."""
    with sqlite3.connect(source_db) as source:
        with sqlite3.connect(backup_db) as backup:
            source.backup(backup)
    print(f"Database backed up to {backup_db}")

def restore_database(backup_db, target_db):
    """Restore database from backup."""
    with sqlite3.connect(backup_db) as backup:
        with sqlite3.connect(target_db) as target:
            backup.backup(target)
    print(f"Database restored from {backup_db}")

# Usage
backup_database('example.db', 'backup.db')
restore_database('backup.db', 'restored.db')
```

## Performance Optimization

### Indexing Strategies

```python
import sqlite3

with sqlite3.connect('example.db') as conn:
    cursor = conn.cursor()
    
    # Create indexes for frequently queried columns
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_date)")
    
    # Composite index for multiple column queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_user_created ON posts(user_id, created_date)")
    
    # Analyze query performance
    cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = ?", ('test@example.com',))
    plan = cursor.fetchall()
    for row in plan:
        print(row)
```

### Batch Operations

```python
import sqlite3

def batch_insert_users(users_data, batch_size=1000):
    """Insert users in batches for better performance."""
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        
        for i in range(0, len(users_data), batch_size):
            batch = users_data[i:i + batch_size]
            cursor.executemany(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                batch
            )
        
        conn.commit()
        print(f"Inserted {len(users_data)} users in batches of {batch_size}")

# Example usage
large_dataset = [
    (f'user_{i}', f'user_{i}@example.com') 
    for i in range(10000)
]
batch_insert_users(large_dataset)
```

### Connection Optimization

```python
import sqlite3

def optimize_connection(conn):
    """Apply performance optimizations to connection."""
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL")
    
    # Increase cache size (in KB)
    cursor.execute("PRAGMA cache_size=10000")
    
    # Set synchronous mode for better performance
    cursor.execute("PRAGMA synchronous=NORMAL")
    
    # Increase memory for temporary tables
    cursor.execute("PRAGMA temp_store=MEMORY")
    
    # Optimize page size
    cursor.execute("PRAGMA page_size=4096")

# Usage
with sqlite3.connect('example.db') as conn:
    optimize_connection(conn)
    # Perform database operations
```

## Security Considerations

### SQL Injection Prevention

```python
import sqlite3

# NEVER do this (vulnerable to SQL injection)
def bad_query(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    # This is dangerous!

# ALWAYS do this (safe from SQL injection)
def safe_query(username):
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchall()

# Using named parameters (also safe)
def safe_named_query(username, email):
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = :username AND email = :email",
            {'username': username, 'email': email}
        )
        return cursor.fetchall()
```

### Input Validation

```python
import sqlite3
import re

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def safe_insert_user(username, email):
    """Safely insert user with validation."""
    # Validate input
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters")
    
    if not validate_email(email):
        raise ValueError("Invalid email format")
    
    # Sanitize input
    username = username.strip()
    email = email.strip().lower()
    
    try:
        with sqlite3.connect('example.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                (username, email)
            )
            return cursor.lastrowid
            
    except sqlite3.IntegrityError:
        raise ValueError("Username or email already exists")
```

## Best Practices

### Database Design Principles

1. **Use appropriate data types**
2. **Implement proper constraints**
3. **Create necessary indexes**
4. **Normalize your schema**
5. **Use foreign keys for referential integrity**

### Code Organization

```python
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

class DatabaseManager:
    """Database manager class for organized SQLite operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database with required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def create_user(self, username: str, email: str) -> int:
        """Create a new user and return the user ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                (username, email)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all users with optional limit."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

# Usage
db = DatabaseManager('example.db')
user_id = db.create_user('john_doe', 'john@example.com')
user = db.get_user(user_id)
all_users = db.get_users()
```

## Examples

### Complete CRUD Application

```python
import sqlite3
import datetime
from typing import List, Dict, Optional

class BlogManager:
    """Complete blog management system using SQLite."""
    
    def __init__(self, db_path: str = 'blog.db'):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Posts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    author_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (author_id) REFERENCES users (id)
                )
            ''')
            
            # Comments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER NOT NULL,
                    author_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts (id),
                    FOREIGN KEY (author_id) REFERENCES users (id)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id)')
            
            conn.commit()
    
    def create_user(self, username: str, email: str, password_hash: str) -> int:
        """Create a new user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            conn.commit()
            return cursor.lastrowid
    
    def create_post(self, title: str, content: str, author_id: int) -> int:
        """Create a new post."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO posts (title, content, author_id) VALUES (?, ?, ?)",
                (title, content, author_id)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_posts_with_authors(self, limit: int = 10) -> List[Dict]:
        """Get posts with author information."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.id, p.title, p.content, p.created_at,
                       u.username, u.email
                FROM posts p
                JOIN users u ON p.author_id = u.id
                ORDER BY p.created_at DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def add_comment(self, post_id: int, author_id: int, content: str) -> int:
        """Add a comment to a post."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO comments (post_id, author_id, content) VALUES (?, ?, ?)",
                (post_id, author_id, content)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_post_with_comments(self, post_id: int) -> Optional[Dict]:
        """Get a post with all its comments."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get post
            cursor.execute('''
                SELECT p.*, u.username as author_username
                FROM posts p
                JOIN users u ON p.author_id = u.id
                WHERE p.id = ?
            ''', (post_id,))
            
            post = cursor.fetchone()
            if not post:
                return None
            
            post_dict = dict(post)
            
            # Get comments
            cursor.execute('''
                SELECT c.*, u.username as author_username
                FROM comments c
                JOIN users u ON c.author_id = u.id
                WHERE c.post_id = ?
                ORDER BY c.created_at ASC
            ''', (post_id,))
            
            comments = [dict(row) for row in cursor.fetchall()]
            post_dict['comments'] = comments
            
            return post_dict

# Usage example
blog = BlogManager()

# Create users
user1_id = blog.create_user('alice', 'alice@example.com', 'hashed_password_1')
user2_id = blog.create_user('bob', 'bob@example.com', 'hashed_password_2')

# Create posts
post_id = blog.create_post('SQLite with Python', 'Great tutorial content...', user1_id)

# Add comments
blog.add_comment(post_id, user2_id, 'Great post!')

# Get post with comments
post_with_comments = blog.get_post_with_comments(post_id)
print(post_with_comments)
```

## API Reference

### sqlite3 Module Functions

#### `connect(database, timeout=5.0, detect_types=0, isolation_level='DEFERRED', check_same_thread=True, factory=Connection, cached_statements=128, uri=False)`
Create a connection to an SQLite database.

**Parameters:**
- `database`: Database file path or `:memory:` for in-memory database
- `timeout`: Connection timeout in seconds
- `detect_types`: Type detection flags
- `isolation_level`: Transaction isolation level
- `check_same_thread`: Thread safety check
- `factory`: Connection factory class
- `cached_statements`: Number of cached statements
- `uri`: Enable URI filenames

#### `register_converter(typename, callable)`
Register a callable to convert a bytestring from the database into a custom Python type.

#### `register_adapter(type, callable)`
Register a callable to convert the custom Python type into one of SQLite's supported types.

### Connection Methods

#### `execute(sql, parameters=())`
Execute an SQL statement with optional parameters.

#### `executemany(sql, seq_of_parameters)`
Execute an SQL statement multiple times with different parameters.

#### `executescript(sql_script)`
Execute multiple SQL statements separated by semicolons.

#### `commit()`
Commit the current transaction.

#### `rollback()`
Roll back the current transaction.

#### `close()`
Close the database connection.

#### `cursor()`
Create a new cursor object.

#### `backup(target, pages=-1, progress=None, name='main', sleep=0.250)`
Create a backup of the database.

### Cursor Methods

#### `execute(sql, parameters=())`
Execute a single SQL statement.

#### `executemany(sql, seq_of_parameters)`
Execute an SQL statement for each set of parameters.

#### `fetchone()`
Fetch the next row of a query result set.

#### `fetchmany(size=cursor.arraysize)`
Fetch the next set of rows of a query result set.

#### `fetchall()`
Fetch all remaining rows of a query result set.

#### `close()`
Close the cursor.

### Properties and Attributes

#### Connection Properties
- `isolation_level`: Get or set the isolation level
- `in_transaction`: True if a transaction is active
- `row_factory`: Row factory function
- `text_factory`: Text factory function
- `total_changes`: Total number of database changes

#### Cursor Properties
- `description`: Column descriptions for the last query
- `lastrowid`: Row ID of the last inserted row
- `rowcount`: Number of rows affected by the last statement
- `arraysize`: Default number of rows to fetch with fetchmany()

### Exception Classes

#### `sqlite3.Error`
Base class for all SQLite exceptions.

#### `sqlite3.Warning`
Exception raised for important warnings.

#### `sqlite3.InterfaceError`
Exception raised for errors related to the database interface.

#### `sqlite3.DatabaseError`
Exception raised for errors related to the database.

#### `sqlite3.DataError`
Exception raised for errors due to problems with the processed data.

#### `sqlite3.OperationalError`
Exception raised for errors related to the database's operation.

#### `sqlite3.IntegrityError`
Exception raised when the relational integrity of the database is affected.

#### `sqlite3.InternalError`
Exception raised when the database encounters an internal error.

#### `sqlite3.ProgrammingError`
Exception raised for programming errors.

#### `sqlite3.NotSupportedError`
Exception raised when a method or database API is not supported.

### Constants

#### `sqlite3.PARSE_DECLTYPES`
Make sqlite3 parse the declared type for each column.

#### `sqlite3.PARSE_COLNAMES`
Make sqlite3 parse the column names for each column.

#### `sqlite3.version`
Version number of this module as a string.

#### `sqlite3.version_info`
Version number of this module as a tuple of integers.

#### `sqlite3.sqlite_version`
Version number of the runtime SQLite library as a string.

#### `sqlite3.sqlite_version_info`
Version number of the runtime SQLite library as a tuple of integers.

## Advanced Use Cases

### Database Schema Management

```python
import sqlite3
from typing import List, Dict

class SchemaManager:
    """Manage database schema versions and migrations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_schema_table()
    
    def init_schema_table(self):
        """Create schema_version table to track migrations."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''')
            conn.commit()
    
    def get_current_version(self) -> int:
        """Get the current schema version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()[0]
            return result if result is not None else 0
    
    def apply_migration(self, version: int, description: str, sql_commands: List[str]):
        """Apply a database migration."""
        current_version = self.get_current_version()
        
        if version <= current_version:
            print(f"Migration {version} already applied")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Execute migration commands
                for command in sql_commands:
                    cursor.execute(command)
                
                # Record migration
                cursor.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (version, description)
                )
                
                conn.commit()
                print(f"Applied migration {version}: {description}")
                
            except sqlite3.Error as e:
                conn.rollback()
                print(f"Failed to apply migration {version}: {e}")
                raise

# Usage example
schema_manager = SchemaManager('example.db')

# Migration 1: Create initial tables
migration_1_commands = [
    '''CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL
    )''',
    '''CREATE INDEX idx_users_email ON users(email)'''
]

schema_manager.apply_migration(1, "Create users table", migration_1_commands)

# Migration 2: Add posts table
migration_2_commands = [
    '''CREATE TABLE posts (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT,
        user_id INTEGER,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )'''
]

schema_manager.apply_migration(2, "Create posts table", migration_2_commands)
```

### Connection Pool Implementation

```python
import sqlite3
import threading
import queue
from contextlib import contextmanager
from typing import Optional

class ConnectionPool:
    """Simple connection pool for SQLite."""
    
    def __init__(self, database: str, max_connections: int = 10):
        self.database = database
        self.max_connections = max_connections
        self.pool = queue.Queue(maxsize=max_connections)
        self.current_connections = 0
        self.lock = threading.Lock()
        
        # Pre-populate pool with some connections
        for _ in range(min(3, max_connections)):
            self._create_connection()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(
            self.database,
            check_same_thread=False,
            timeout=30.0
        )
        conn.row_factory = sqlite3.Row
        
        # Apply optimizations
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA foreign_keys=ON")
        
        with self.lock:
            self.current_connections += 1
        
        return conn
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        try:
            # Try to get an existing connection
            return self.pool.get_nowait()
        except queue.Empty:
            # Create new connection if under limit
            with self.lock:
                if self.current_connections < self.max_connections:
                    return self._create_connection()
            
            # Wait for available connection
            return self.pool.get(timeout=10.0)
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        if conn and not self.pool.full():
            self.pool.put(conn)
        else:
            # Close excess connections
            if conn:
                conn.close()
                with self.lock:
                    self.current_connections -= 1
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for getting database connections."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)
    
    def close_all(self):
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        with self.lock:
            self.current_connections = 0

# Usage example
pool = ConnectionPool('example.db', max_connections=5)

def database_worker(worker_id: int):
    """Example worker function using connection pool."""
    with pool.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email) VALUES (?, ?)",
            (f'worker_{worker_id}', f'worker_{worker_id}@example.com')
        )
        conn.commit()
        print(f"Worker {worker_id} completed database operation")

# Use in multiple threads
import threading

threads = []
for i in range(10):
    thread = threading.Thread(target=database_worker, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

pool.close_all()
```

### Full-Text Search Implementation

```python
import sqlite3
from typing import List, Dict, Optional

class FullTextSearchManager:
    """Implement full-text search using SQLite FTS5."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_fts_tables()
    
    def setup_fts_tables(self):
        """Create FTS virtual tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create regular articles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    author TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create FTS virtual table
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                    title, content, author,
                    content='articles',
                    content_rowid='id'
                )
            ''')
            
            # Create triggers to keep FTS table synchronized
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
                    INSERT INTO articles_fts(rowid, title, content, author)
                    VALUES (new.id, new.title, new.content, new.author);
                END
            ''')
            
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
                    INSERT INTO articles_fts(articles_fts, rowid, title, content, author)
                    VALUES('delete', old.id, old.title, old.content, old.author);
                END
            ''')
            
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
                    INSERT INTO articles_fts(articles_fts, rowid, title, content, author)
                    VALUES('delete', old.id, old.title, old.content, old.author);
                    INSERT INTO articles_fts(rowid, title, content, author)
                    VALUES (new.id, new.title, new.content, new.author);
                END
            ''')
            
            conn.commit()
    
    def add_article(self, title: str, content: str, author: str = None) -> int:
        """Add a new article."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO articles (title, content, author) VALUES (?, ?, ?)",
                (title, content, author)
            )
            conn.commit()
            return cursor.lastrowid
    
    def search_articles(self, query: str, limit: int = 10) -> List[Dict]:
        """Search articles using full-text search."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Use FTS5 MATCH operator for full-text search
            cursor.execute('''
                SELECT a.id, a.title, a.content, a.author, a.created_at,
                       rank
                FROM articles_fts
                JOIN articles a ON articles_fts.rowid = a.id
                WHERE articles_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (query, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def search_with_snippets(self, query: str, limit: int = 10) -> List[Dict]:
        """Search articles and return snippets with highlighted matches."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT a.id, a.title, a.author, a.created_at,
                       snippet(articles_fts, 1, '<mark>', '</mark>', '...', 64) as snippet
                FROM articles_fts
                JOIN articles a ON articles_fts.rowid = a.id
                WHERE articles_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (query, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def search_by_field(self, field: str, query: str, limit: int = 10) -> List[Dict]:
        """Search in a specific field (title, content, or author)."""
        if field not in ['title', 'content', 'author']:
            raise ValueError("Field must be 'title', 'content', or 'author'")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            fts_query = f"{field}:{query}"
            cursor.execute('''
                SELECT a.id, a.title, a.content, a.author, a.created_at
                FROM articles_fts
                JOIN articles a ON articles_fts.rowid = a.id
                WHERE articles_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (fts_query, limit))
            
            return [dict(row) for row in cursor.fetchall()]

# Usage example
fts_manager = FullTextSearchManager('fts_example.db')

# Add some articles
fts_manager.add_article(
    "SQLite Full-Text Search",
    "SQLite provides powerful full-text search capabilities through FTS5 virtual tables...",
    "John Doe"
)

fts_manager.add_article(
    "Python Database Programming",
    "Learn how to use Python with SQLite for efficient database operations...",
    "Jane Smith"
)

fts_manager.add_article(
    "Advanced SQLite Features",
    "Explore advanced SQLite features including triggers, views, and JSON support...",
    "Bob Johnson"
)

# Search examples
search_results = fts_manager.search_articles("SQLite Python")
print("Search results:", search_results)

snippet_results = fts_manager.search_with_snippets("database")
print("Snippet results:", snippet_results)

title_results = fts_manager.search_by_field("title", "SQLite")
print("Title search results:", title_results)
```

## Troubleshooting

### Common Issues and Solutions

#### Database Locked Error

```python
import sqlite3
import time
import random

def retry_database_operation(func, max_retries=5, base_delay=0.1):
    """Retry database operation with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue
            raise
    
    raise sqlite3.OperationalError("Max retries exceeded")

# Usage
def database_operation():
    with sqlite3.connect('example.db', timeout=30.0) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)",
                      ('test_user', 'test@example.com'))
        conn.commit()
        return cursor.lastrowid

try:
    user_id = retry_database_operation(database_operation)
    print(f"User created with ID: {user_id}")
except sqlite3.OperationalError as e:
    print(f"Database operation failed: {e}")
```

#### Memory Usage Optimization

```python
import sqlite3

def process_large_dataset(db_path: str, batch_size: int = 1000):
    """Process large datasets efficiently with memory management."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Use server-side cursor for large result sets
        cursor.execute("SELECT id, data FROM large_table ORDER BY id")
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            # Process batch
            for row in rows:
                process_row(row)
            
            # Optional: force garbage collection for very large datasets
            import gc
            gc.collect()

def process_row(row):
    """Process individual row."""
    # Your processing logic here
    pass
```

#### Debugging SQL Queries

```python
import sqlite3
import logging

# Enable SQLite debugging
logging.basicConfig(level=logging.DEBUG)

class DebuggingConnection(sqlite3.Connection):
    """Connection class that logs all SQL operations."""
    
    def execute(self, sql, parameters=()):
        logging.debug(f"Executing: {sql}")
        if parameters:
            logging.debug(f"Parameters: {parameters}")
        return super().execute(sql, parameters)
    
    def executemany(self, sql, seq_of_parameters):
        logging.debug(f"Executing many: {sql}")
        logging.debug(f"Parameter count: {len(list(seq_of_parameters))}")
        return super().executemany(sql, seq_of_parameters)

# Use debugging connection
conn = sqlite3.connect('example.db', factory=DebuggingConnection)
cursor = conn.cursor()
cursor.execute("SELECT * FROM users WHERE username = ?", ('john_doe',))
results = cursor.fetchall()
conn.close()
```

## Performance Benchmarking

```python
import sqlite3
import time
import contextlib
from typing import Callable, Any

@contextlib.contextmanager
def timing_context(description: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{description}: {end_time - start_time:.4f} seconds")

def benchmark_operations():
    """Benchmark various SQLite operations."""
    
    # Setup test database
    with sqlite3.connect(':memory:') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value INTEGER
            )
        ''')
        
        # Benchmark single inserts
        with timing_context("1000 individual inserts"):
            for i in range(1000):
                cursor.execute("INSERT INTO test_table (name, value) VALUES (?, ?)",
                             (f"name_{i}", i))
            conn.commit()
        
        # Clear table
        cursor.execute("DELETE FROM test_table")
        
        # Benchmark batch inserts
        test_data = [(f"name_{i}", i) for i in range(1000)]
        with timing_context("1000 batch inserts"):
            cursor.executemany("INSERT INTO test_table (name, value) VALUES (?, ?)",
                             test_data)
            conn.commit()
        
        # Benchmark selects
        with timing_context("1000 individual selects"):
            for i in range(0, 1000, 10):
                cursor.execute("SELECT * FROM test_table WHERE id = ?", (i,))
                cursor.fetchone()
        
        # Benchmark bulk select
        with timing_context("Bulk select all records"):
            cursor.execute("SELECT * FROM test_table")
            cursor.fetchall()
        
        # Create index and benchmark
        cursor.execute("CREATE INDEX idx_value ON test_table(value)")
        with timing_context("Indexed query"):
            cursor.execute("SELECT * FROM test_table WHERE value > 500")
            cursor.fetchall()

if __name__ == "__main__":
    benchmark_operations()
```

## Conclusion

This comprehensive guide covers the essential aspects of using SQLite with Python. Key takeaways include:

1. **Always use parameterized queries** to prevent SQL injection
2. **Use context managers** for proper resource management
3. **Implement proper error handling** for robust applications
4. **Consider performance optimizations** for large datasets
5. **Follow best practices** for maintainable code
6. **Use appropriate data types and constraints** for data integrity
7. **Implement proper indexing** for query performance

SQLite with Python provides a powerful, lightweight solution for many database needs, from simple scripts to complex applications. The combination of SQLite's simplicity and Python's expressiveness makes it an excellent choice for rapid development and deployment.

Remember to always test your database operations thoroughly, especially when dealing with concurrent access or large datasets. Consider using connection pooling for multi-threaded applications and implement proper backup strategies for production systems.

For more advanced use cases, explore SQLite's extensive documentation and consider complementary tools like SQLAlchemy for ORM functionality or additional Python packages for specific database management needs.
