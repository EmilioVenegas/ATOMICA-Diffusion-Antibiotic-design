#!/bin/bash
# Script to restore ChEMBL PostgreSQL database from tar.gz dump file

set -e

# Default values
DUMP_FILE="${1:-chembl_36_postgresql.tar.gz}"
DB_NAME="${2:-chembl_36}"
DB_USER="${3:-$USER}"
DB_HOST="${4:-localhost}"
DB_PORT="${5:-5432}"

echo "=========================================="
echo "ChEMBL PostgreSQL Database Restore"
echo "=========================================="
echo "Dump file: $DUMP_FILE"
echo "Database name: $DB_NAME"
echo "User: $DB_USER"
echo "Host: $DB_HOST"
echo "Port: $DB_PORT"
echo ""

# Check if dump file exists
if [ ! -f "$DUMP_FILE" ]; then
    echo "Error: Dump file not found: $DUMP_FILE"
    echo "Usage: $0 <dump_file.tar.gz> [db_name] [db_user] [db_host] [db_port]"
    exit 1
fi

# Check if PostgreSQL is running
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" > /dev/null 2>&1; then
    echo "Error: Cannot connect to PostgreSQL server at $DB_HOST:$DB_PORT"
    echo "Please ensure PostgreSQL is running and accessible."
    exit 1
fi

echo "Step 1: Extracting tar.gz file..."
TEMP_DIR=$(mktemp -d)
tar -xzf "$DUMP_FILE" -C "$TEMP_DIR"
echo "✓ Extracted to $TEMP_DIR"

# Find the actual dump file (could be .sql, .dump, .dmp, or .tar)
DUMP_SQL=$(find "$TEMP_DIR" -name "*.sql" | head -1)
DUMP_DMP=$(find "$TEMP_DIR" -name "*.dmp" | head -1)
DUMP_TAR=$(find "$TEMP_DIR" -name "*.tar" | head -1)

if [ -z "$DUMP_SQL" ] && [ -z "$DUMP_DMP" ] && [ -z "$DUMP_TAR" ]; then
    echo "Error: Could not find dump file in archive"
    echo "Contents of extracted directory:"
    find "$TEMP_DIR" -type f | head -10
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo ""
echo "Step 2: Dropping existing database (if exists)..."
dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME" || true
echo "✓ Database dropped (if it existed)"

echo ""
echo "Step 3: Creating new database..."
createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
echo "✓ Database created"

echo ""
echo "Step 4: Restoring database..."
echo "This may take a while (10-30 minutes depending on database size)..."

if [ -n "$DUMP_SQL" ]; then
    echo "Restoring from SQL dump: $DUMP_SQL"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$DUMP_SQL"
elif [ -n "$DUMP_DMP" ]; then
    echo "Restoring from PostgreSQL custom dump: $DUMP_DMP"
    pg_restore -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" --no-owner --no-acl --verbose "$DUMP_DMP"
elif [ -n "$DUMP_TAR" ]; then
    echo "Restoring from tar dump: $DUMP_TAR"
    pg_restore -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" "$DUMP_TAR"
fi

echo ""
echo "✓ Database restored successfully!"

echo ""
echo "Step 5: Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo "✓ Cleanup complete"

echo ""
echo "=========================================="
echo "Database '$DB_NAME' is ready to use!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python 3pbq_evaluation/generate_decoys_from_postgres.py \\"
echo "    --dbname $DB_NAME \\"
echo "    --user $DB_USER \\"
echo "    --host $DB_HOST \\"
echo "    --port $DB_PORT \\"
echo "    --output 3pbq_evaluation/inputs/3pbq_decoys.csv \\"
echo "    --n-samples 50"

