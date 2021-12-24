# Predict Future Sales
![](assets/kaggle_screenshot.png)
### This repository is an entry to the [Predict Future Sales Data Science Competition](https://www.kaggle.com/c/competitive-data-science-predict-future-sales) on Kaggle. The challenge is to work with a time-series dataset, consisting of daily sales data provided by one of the largest Russian software firms.

&nbsp;  
## The Data consists of these Datapoints:
1. shop_id - unique identifier of a shop
2. item_id - unique identifier of an item
3. item_category_id - category group that the item is in
4. item_cnt_day - the amount of sales of the item at that day in that shop
5. item_price - price of the item
6. date - date of entry
7. date_block_num - consecutive month number, January 2013 is 0, February 2013 is 1 etc.
8. item_name - name of item
9. shop_name name of shop
10. item_category_name - name of item_category
&nbsp;

### It contains data from January 2013 all the way to October 2015, with a task to predict the sales in the month of November 2015.
I cleaned the data and used it to build a model in LightGBM, for doing exactly that. Independently of the Model I set the prediction to 0 for all Item/Shop Combinations, if the item was not sold in the given Shop in the last 3 Months 
The model has an Accuracy of 1.15702.
