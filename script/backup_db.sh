#!/bin/bash
# Automated PostgreSQL backup script for NGLab
# Performs pg_dump and uploads to S3/GCS with retention policy

set -e

# Configuration
BACKUP_DIR="/tmp/nglab_backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="nglab_backup_${TIMESTAMP}.sql.gz"
DB_NAME="${POSTGRES_DB:-nglab}"
DB_USER="${POSTGRES_USER:-nglab}"
DB_HOST="${POSTGRES_HOST:-postgres}"

# Retention settings
DAILY_RETENTION=7
WEEKLY_RETENTION=4
MONTHLY_RETENTION=12

# Create backup directory
mkdir -p "${BACKUP_DIR}"

echo "Starting backup: ${BACKUP_FILE}"

# Perform backup with compression
PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
    -h "${DB_HOST}" \
    -U "${DB_USER}" \
    -d "${DB_NAME}" \
    --verbose \
    --format=plain \
    | gzip > "${BACKUP_DIR}/${BACKUP_FILE}"

BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
echo "Backup complete: ${BACKUP_SIZE}"

# Upload to S3 (if configured)
if [ -n "${S3_BACKUP_BUCKET}" ]; then
    echo "Uploading to S3: s3://${S3_BACKUP_BUCKET}/backups/${BACKUP_FILE}"
    aws s3 cp \
        "${BACKUP_DIR}/${BACKUP_FILE}" \
        "s3://${S3_BACKUP_BUCKET}/backups/${BACKUP_FILE}" \
        --storage-class STANDARD_IA
    
    echo "S3 upload complete"
fi

# Upload to GCS (if configured)
if [ -n "${GCS_BACKUP_BUCKET}" ]; then
    echo "Uploading to GCS: gs://${GCS_BACKUP_BUCKET}/backups/${BACKUP_FILE}"
    gsutil cp \
        "${BACKUP_DIR}/${BACKUP_FILE}" \
        "gs://${GCS_BACKUP_BUCKET}/backups/${BACKUP_FILE}"
    
    echo "GCS upload complete"
fi

# Apply retention policy
echo "Applying retention policy..."

# Keep daily backups for last 7 days
find "${BACKUP_DIR}" -name "nglab_backup_*.sql.gz" -mtime +${DAILY_RETENTION} -delete

# Keep weekly backups (Sundays) for 4 weeks
# Keep monthly backups (1st of month) for 12 months

# Cleanup local backup
rm -f "${BACKUP_DIR}/${BACKUP_FILE}"

echo "Backup process complete"
