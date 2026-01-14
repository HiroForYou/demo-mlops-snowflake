-- ===============================================================================
-- ARCA BEVERAGE DEMO: DATA SETUP
-- ===============================================================================
-- Creates realistic dummy data for retail beverage forecasting demo
-- Industry: Retail Beverage Sales
-- Use Case: Weekly sales forecast with customer segmentation
-- Volume: ~100K transactions, 2 years history + 4 weeks inference
-- ===============================================================================

-- Setup database and schema
CREATE OR REPLACE DATABASE ARCA_BEVERAGE_DEMO;
USE DATABASE ARCA_BEVERAGE_DEMO;
CREATE OR REPLACE SCHEMA ML_DATA;
USE SCHEMA ML_DATA;

-- Create warehouse for demo
CREATE OR REPLACE WAREHOUSE ARCA_DEMO_WH 
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE;

USE WAREHOUSE ARCA_DEMO_WH;

-- ===============================================================================
-- 1. CUSTOMERS TABLE
-- ===============================================================================
CREATE OR REPLACE TABLE CUSTOMERS (
    CUSTOMER_ID NUMBER PRIMARY KEY,
    SEGMENT VARCHAR(50),
    REGISTRATION_DATE DATE,
    LOCATION VARCHAR(100),
    CUSTOMER_NAME VARCHAR(200),
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Generate 1000 customers across different locations
INSERT INTO CUSTOMERS (CUSTOMER_ID, SEGMENT, REGISTRATION_DATE, LOCATION, CUSTOMER_NAME)
SELECT
    SEQ4() AS CUSTOMER_ID,
    CASE 
        WHEN UNIFORM(1, 100, RANDOM()) <= 20 THEN 'SEGMENT_1'
        WHEN UNIFORM(1, 100, RANDOM()) <= 40 THEN 'SEGMENT_2'
        WHEN UNIFORM(1, 100, RANDOM()) <= 60 THEN 'SEGMENT_3'
        WHEN UNIFORM(1, 100, RANDOM()) <= 75 THEN 'SEGMENT_4'
        WHEN UNIFORM(1, 100, RANDOM()) <= 90 THEN 'SEGMENT_5'
        ELSE 'SEGMENT_6'
    END AS SEGMENT,
    DATEADD(DAY, -UNIFORM(100, 730, RANDOM()), CURRENT_DATE()) AS REGISTRATION_DATE,
    CASE UNIFORM(1, 5, RANDOM())
        WHEN 1 THEN 'Mexico City'
        WHEN 2 THEN 'Guadalajara'
        WHEN 3 THEN 'Monterrey'
        WHEN 4 THEN 'Puebla'
        ELSE 'Tijuana'
    END AS LOCATION,
    'Customer_' || SEQ4() AS CUSTOMER_NAME
FROM TABLE(GENERATOR(ROWCOUNT => 1000));

-- ===============================================================================
-- 2. PRODUCTS TABLE
-- ===============================================================================
CREATE OR REPLACE TABLE PRODUCTS (
    PRODUCT_ID NUMBER PRIMARY KEY,
    PRODUCT_NAME VARCHAR(200),
    CATEGORY VARCHAR(100),
    PRICE NUMBER(10,2),
    BRAND VARCHAR(100),
    SIZE_ML NUMBER,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Generate 50 beverage products
INSERT INTO PRODUCTS (PRODUCT_ID, PRODUCT_NAME, CATEGORY, PRICE, BRAND, SIZE_ML)
VALUES
    (1, 'Coca-Cola Classic', 'Carbonated Soft Drink', 15.50, 'Coca-Cola', 500),
    (2, 'Coca-Cola Light', 'Carbonated Soft Drink', 15.50, 'Coca-Cola', 500),
    (3, 'Coca-Cola Zero', 'Carbonated Soft Drink', 15.50, 'Coca-Cola', 500),
    (4, 'Sprite', 'Carbonated Soft Drink', 14.50, 'Coca-Cola', 500),
    (5, 'Fanta Orange', 'Carbonated Soft Drink', 14.50, 'Coca-Cola', 500),
    (6, 'Powerade Mountain Blast', 'Sports Drink', 18.00, 'Coca-Cola', 500),
    (7, 'Powerade Fruit Punch', 'Sports Drink', 18.00, 'Coca-Cola', 500),
    (8, 'Dasani Water', 'Water', 12.00, 'Coca-Cola', 600),
    (9, 'Ciel Water', 'Water', 10.00, 'Coca-Cola', 600),
    (10, 'Del Valle Orange Juice', 'Juice', 22.00, 'Coca-Cola', 500),
    (11, 'Del Valle Apple Juice', 'Juice', 22.00, 'Coca-Cola', 500),
    (12, 'Minute Maid Orange', 'Juice', 20.00, 'Coca-Cola', 500),
    (13, 'Fuze Tea Lemon', 'Tea', 16.00, 'Coca-Cola', 500),
    (14, 'Fuze Tea Peach', 'Tea', 16.00, 'Coca-Cola', 500),
    (15, 'Coca-Cola 2L', 'Carbonated Soft Drink', 35.00, 'Coca-Cola', 2000),
    (16, 'Sprite 2L', 'Carbonated Soft Drink', 33.00, 'Coca-Cola', 2000),
    (17, 'Fanta 2L', 'Carbonated Soft Drink', 33.00, 'Coca-Cola', 2000),
    (18, 'Smartwater', 'Water', 25.00, 'Coca-Cola', 700),
    (19, 'Honest Tea Organic', 'Tea', 24.00, 'Coca-Cola', 500),
    (20, 'Simply Orange Juice', 'Juice', 28.00, 'Coca-Cola', 500),
    (21, 'Coca-Cola Energy', 'Energy Drink', 30.00, 'Coca-Cola', 250),
    (22, 'Vitamin Water XXX', 'Enhanced Water', 20.00, 'Coca-Cola', 500),
    (23, 'Fresca', 'Carbonated Soft Drink', 14.00, 'Coca-Cola', 500),
    (24, 'Mello Yello', 'Carbonated Soft Drink', 15.00, 'Coca-Cola', 500),
    (25, 'Barqs Root Beer', 'Carbonated Soft Drink', 15.00, 'Coca-Cola', 500),
    (26, 'Coca-Cola Cherry', 'Carbonated Soft Drink', 16.00, 'Coca-Cola', 500),
    (27, 'Sprite Zero', 'Carbonated Soft Drink', 14.50, 'Coca-Cola', 500),
    (28, 'Powerade Zero', 'Sports Drink', 18.00, 'Coca-Cola', 500),
    (29, 'Gold Peak Tea', 'Tea', 18.00, 'Coca-Cola', 500),
    (30, 'Topo Chico Mineral Water', 'Water', 22.00, 'Coca-Cola', 355),
    (31, 'Topo Chico Hard Seltzer', 'Alcoholic Beverage', 35.00, 'Coca-Cola', 355),
    (32, 'AdeS Soy Milk', 'Plant-Based', 24.00, 'Coca-Cola', 1000),
    (33, 'Fairlife Milk', 'Dairy', 28.00, 'Coca-Cola', 500),
    (34, 'Core Power Protein Shake', 'Protein Drink', 35.00, 'Coca-Cola', 414),
    (35, 'Peace Tea', 'Tea', 14.00, 'Coca-Cola', 695),
    (36, 'Odwalla Smoothie', 'Smoothie', 32.00, 'Coca-Cola', 450),
    (37, 'Zico Coconut Water', 'Coconut Water', 26.00, 'Coca-Cola', 500),
    (38, 'Innocent Smoothie', 'Smoothie', 30.00, 'Coca-Cola', 360),
    (39, 'Costa Coffee RTD', 'Coffee', 28.00, 'Coca-Cola', 250),
    (40, 'Coca-Cola Coffee', 'Coffee', 25.00, 'Coca-Cola', 250),
    (41, 'Schweppes Ginger Ale', 'Carbonated Soft Drink', 15.00, 'Coca-Cola', 500),
    (42, 'Schweppes Tonic Water', 'Mixer', 15.00, 'Coca-Cola', 500),
    (43, 'Coca-Cola Life', 'Carbonated Soft Drink', 16.00, 'Coca-Cola', 500),
    (44, 'Fanta Grape', 'Carbonated Soft Drink', 14.50, 'Coca-Cola', 500),
    (45, 'Fanta Pineapple', 'Carbonated Soft Drink', 14.50, 'Coca-Cola', 500),
    (46, 'Aquarius Sports Drink', 'Sports Drink', 17.00, 'Coca-Cola', 500),
    (47, 'Kin Tea', 'Tea', 15.00, 'Coca-Cola', 500),
    (48, 'Lift Energy', 'Energy Drink', 28.00, 'Coca-Cola', 250),
    (49, 'Glaceau Vitaminwater', 'Enhanced Water', 20.00, 'Coca-Cola', 500),
    (50, 'Dasani Sparkling', 'Sparkling Water', 14.00, 'Coca-Cola', 500);

-- ===============================================================================
-- 3. TRANSACTIONS TABLE (Historical + Inference)
-- ===============================================================================
CREATE OR REPLACE TABLE TRANSACTIONS (
    TRANSACTION_ID NUMBER PRIMARY KEY,
    CUSTOMER_ID NUMBER,
    PRODUCT_ID NUMBER,
    TRANSACTION_DATE DATE,
    UNITS_SOLD NUMBER,
    REVENUE NUMBER(12,2),
    STORE_ID NUMBER,
    REGION VARCHAR(100),
    IS_INFERENCE BOOLEAN DEFAULT FALSE,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (CUSTOMER_ID) REFERENCES CUSTOMERS(CUSTOMER_ID),
    FOREIGN KEY (PRODUCT_ID) REFERENCES PRODUCTS(PRODUCT_ID)
);

-- Generate ~100K historical transactions (2 years)
INSERT INTO TRANSACTIONS (TRANSACTION_ID, CUSTOMER_ID, PRODUCT_ID, TRANSACTION_DATE, UNITS_SOLD, REVENUE, STORE_ID, REGION, IS_INFERENCE)
SELECT
    SEQ8() AS TRANSACTION_ID,
    UNIFORM(1, 1000, RANDOM()) AS CUSTOMER_ID,
    UNIFORM(1, 50, RANDOM()) AS PRODUCT_ID,
    DATEADD(DAY, -UNIFORM(1, 730, RANDOM()), CURRENT_DATE() - 28) AS TRANSACTION_DATE,
    UNIFORM(1, 20, RANDOM()) AS UNITS_SOLD,
    0 AS REVENUE,
    UNIFORM(1, 50, RANDOM()) AS STORE_ID,
    CASE UNIFORM(1, 5, RANDOM())
        WHEN 1 THEN 'Mexico City'
        WHEN 2 THEN 'Guadalajara'
        WHEN 3 THEN 'Monterrey'
        WHEN 4 THEN 'Puebla'
        ELSE 'Tijuana'
    END AS REGION,
    FALSE AS IS_INFERENCE
FROM TABLE(GENERATOR(ROWCOUNT => 100000));

-- Calculate revenue based on units and product price
UPDATE TRANSACTIONS t
SET REVENUE = t.UNITS_SOLD * p.PRICE
FROM PRODUCTS p
WHERE t.PRODUCT_ID = p.PRODUCT_ID;

-- Generate inference period transactions (last 4 weeks)
INSERT INTO TRANSACTIONS (TRANSACTION_ID, CUSTOMER_ID, PRODUCT_ID, TRANSACTION_DATE, UNITS_SOLD, REVENUE, STORE_ID, REGION, IS_INFERENCE)
SELECT
    100000 + SEQ8() AS TRANSACTION_ID,
    UNIFORM(1, 1000, RANDOM()) AS CUSTOMER_ID,
    UNIFORM(1, 50, RANDOM()) AS PRODUCT_ID,
    DATEADD(DAY, -UNIFORM(0, 27, RANDOM()), CURRENT_DATE()) AS TRANSACTION_DATE,
    UNIFORM(1, 15, RANDOM()) AS UNITS_SOLD,
    0 AS REVENUE,
    UNIFORM(1, 50, RANDOM()) AS STORE_ID,
    CASE UNIFORM(1, 5, RANDOM())
        WHEN 1 THEN 'Mexico City'
        WHEN 2 THEN 'Guadalajara'
        WHEN 3 THEN 'Monterrey'
        WHEN 4 THEN 'Puebla'
        ELSE 'Tijuana'
    END AS REGION,
    TRUE AS IS_INFERENCE
FROM TABLE(GENERATOR(ROWCOUNT => 5000));

-- Calculate revenue for inference period
UPDATE TRANSACTIONS t
SET REVENUE = t.UNITS_SOLD * p.PRICE
FROM PRODUCTS p
WHERE t.PRODUCT_ID = p.PRODUCT_ID
  AND t.IS_INFERENCE = TRUE;

-- ===============================================================================
-- 4. AGGREGATED WEEKLY SALES VIEW (Training Target)
-- ===============================================================================
CREATE OR REPLACE VIEW WEEKLY_SALES_AGGREGATED AS
SELECT
    CUSTOMER_ID,
    DATE_TRUNC('WEEK', TRANSACTION_DATE) AS WEEK_START_DATE,
    SUM(UNITS_SOLD) AS WEEKLY_SALES_UNITS,
    SUM(REVENUE) AS WEEKLY_SALES_REVENUE,
    COUNT(DISTINCT TRANSACTION_ID) AS TRANSACTION_COUNT,
    COUNT(DISTINCT PRODUCT_ID) AS UNIQUE_PRODUCTS_PURCHASED,
    AVG(UNITS_SOLD) AS AVG_UNITS_PER_TRANSACTION,
    IS_INFERENCE
FROM TRANSACTIONS
GROUP BY CUSTOMER_ID, DATE_TRUNC('WEEK', TRANSACTION_DATE), IS_INFERENCE;

-- ===============================================================================
-- 5. DATA VALIDATION QUERIES
-- ===============================================================================
-- Check data volumes
SELECT 'Customers' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM CUSTOMERS
UNION ALL
SELECT 'Products' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM PRODUCTS
UNION ALL
SELECT 'Transactions (Historical)' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM TRANSACTIONS WHERE IS_INFERENCE = FALSE
UNION ALL
SELECT 'Transactions (Inference)' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM TRANSACTIONS WHERE IS_INFERENCE = TRUE
UNION ALL
SELECT 'Weekly Sales Records' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM WEEKLY_SALES_AGGREGATED;

-- Check date ranges
SELECT
    'Historical Period' AS PERIOD_TYPE,
    MIN(TRANSACTION_DATE) AS START_DATE,
    MAX(TRANSACTION_DATE) AS END_DATE,
    DATEDIFF(DAY, MIN(TRANSACTION_DATE), MAX(TRANSACTION_DATE)) AS DAYS_SPAN
FROM TRANSACTIONS
WHERE IS_INFERENCE = FALSE
UNION ALL
SELECT
    'Inference Period' AS PERIOD_TYPE,
    MIN(TRANSACTION_DATE) AS START_DATE,
    MAX(TRANSACTION_DATE) AS END_DATE,
    DATEDIFF(DAY, MIN(TRANSACTION_DATE), MAX(TRANSACTION_DATE)) AS DAYS_SPAN
FROM TRANSACTIONS
WHERE IS_INFERENCE = TRUE;

-- Check customer distribution
SELECT
    SEGMENT,
    COUNT(*) AS CUSTOMER_COUNT,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS PERCENTAGE
FROM CUSTOMERS
GROUP BY SEGMENT
ORDER BY SEGMENT;

-- Check product category distribution
SELECT
    CATEGORY,
    COUNT(*) AS PRODUCT_COUNT,
    AVG(PRICE) AS AVG_PRICE,
    SUM(t.UNITS_SOLD) AS TOTAL_UNITS_SOLD
FROM PRODUCTS p
LEFT JOIN TRANSACTIONS t ON p.PRODUCT_ID = t.PRODUCT_ID
GROUP BY CATEGORY
ORDER BY TOTAL_UNITS_SOLD DESC;

-- Check weekly sales summary
SELECT
    YEAR(WEEK_START_DATE) AS YEAR,
    COUNT(DISTINCT WEEK_START_DATE) AS WEEKS_COUNT,
    COUNT(DISTINCT CUSTOMER_ID) AS ACTIVE_CUSTOMERS,
    SUM(WEEKLY_SALES_UNITS) AS TOTAL_UNITS,
    SUM(WEEKLY_SALES_REVENUE) AS TOTAL_REVENUE,
    AVG(WEEKLY_SALES_UNITS) AS AVG_WEEKLY_UNITS
FROM WEEKLY_SALES_AGGREGATED
GROUP BY YEAR(WEEK_START_DATE)
ORDER BY YEAR;

-- ===============================================================================
-- SETUP COMPLETE
-- ===============================================================================
SELECT 'Data setup complete! Ready for Feature Store implementation.' AS STATUS;
