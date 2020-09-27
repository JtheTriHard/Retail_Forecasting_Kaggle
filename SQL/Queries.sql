USE KaggleRetail;

SET SQL_SAFE_UPDATES = 0;
DELETE FROM Sales_Train WHERE item_price IS NULL OR item_cnt IS NULL;

/* Aggregate monthly sales per shop-item pair*/
SELECT train.cur_month, train.shop_ID, train.item_ID, items.item_name, items.item_category_ID,
	SUM(train.item_price * train.item_cnt) AS sales
FROM sales_train train
LEFT JOIN items
	ON train.item_ID = items.item_ID
GROUP BY train.cur_month, train.shop_ID, train.item_ID
ORDER BY train.shop_ID, train.cur_month, train.item_ID;

/* Aggregate monthly sales per shop*/
SELECT train.cur_month, train.shop_ID, shops.shop_name,
	SUM(train.item_price * train.item_cnt) AS sales
FROM sales_train train
LEFT JOIN shops
	ON train.shop_ID = shops.shop_ID
GROUP BY train.cur_month, train.shop_ID
ORDER BY train.shop_ID, train.cur_month;

/* Aggregate monthly sales per item*/
SELECT train.cur_month, train.item_ID, items.item_name, items.item_category_ID,
	SUM(train.item_price * train.item_cnt) AS sales
FROM sales_train train
LEFT JOIN items
	ON train.item_ID = items.item_ID
GROUP BY train.cur_month, train.item_ID
ORDER BY train.item_ID, train.cur_month;

/* Aggregate total monthly sales */
SELECT train.cur_month,
	SUM(train.item_price * train.item_cnt) AS sales
FROM sales_train train
GROUP BY train.cur_month
ORDER BY train.cur_month;
