-- Fix for FIR status column: Hibernate may have created a CHECK constraint or
-- PostgreSQL ENUM type that only allows the original enum values.
-- After adding CHARGE_SHEET_SUBMITTED, CHARGE_SHEET_APPROVED, CHARGE_SHEET_FILED
-- to FirStatus, the DB needs to be updated.

-- Step 1: Convert fir.status column to plain varchar(50) to remove any enum/check constraint
ALTER TABLE fir ALTER COLUMN status TYPE varchar(50);

-- Step 2: Convert charge_sheet status and report_type columns to plain varchar(50) (defensive)
ALTER TABLE charge_sheet ALTER COLUMN status TYPE varchar(50);
ALTER TABLE charge_sheet ALTER COLUMN report_type TYPE varchar(50);

-- Step 3: Delete duplicate DRAFT charge sheets (keep only the oldest per FIR)
DELETE FROM charge_sheet
WHERE charge_sheet_id NOT IN (
    SELECT MIN(charge_sheet_id)
    FROM charge_sheet
    WHERE status = 'DRAFT'
    GROUP BY fir_id
)
AND status = 'DRAFT';
