UPDATE Orders
SET
    order_status = 'Dispatched',
    last_updated = CURRENT_TIMESTAMP
WHERE tracking_number = 'JD100003';