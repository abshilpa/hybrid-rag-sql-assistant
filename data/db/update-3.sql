UPDATE Orders
SET
    order_status = 'Delivered',
    expected_delivery = '2026-07-28',
    last_updated = CURRENT_TIMESTAMP
WHERE tracking_number = 'JD100001';