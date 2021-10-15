from itertools import product
from tqdm import tqdm
import pandas as pd
from deep_translator import GoogleTranslator


tqdm.pandas()

# set to true when translating category names
translate_category_names = False
translate_shop_names = False


# load data files
train_data = pd.read_csv("data/sales_train.csv")
item_data = pd.read_csv("data/items.csv")
category_names = pd.read_csv("data/item_categories.csv")
item_names = pd.read_csv("data/items.csv")
shop_names = pd.read_csv("data/shops.csv")
prediction_data = pd.read_csv("data/test.csv")

# translate category names using google translate
if translate_category_names:
    translator = GoogleTranslator(source='ru', target='en')
    category_names["item_category_name"] = category_names["item_category_name"].progress_apply(lambda x: translator.translate(x))
    category_names.to_csv("data/item_categories.csv", index = False)

# translate shop names using google translate
if translate_shop_names:
    translator = GoogleTranslator(source='ru', target='en')
    shop_names["shop_name"] = shop_names["shop_name"].progress_apply(lambda x: translator.translate(x))
    shop_names.to_csv("data/shops.csv", index = False)


# convert price to euros
train_data["item_price"] = train_data["item_price"] * 0.012
train_data["item_price"] = train_data["item_price"].round(2)

# drop items with count of < 0 since they are probably refunds
train_data = train_data[train_data["item_cnt_day"] > 0]

# filter out outliers
train_data = train_data[train_data["item_price"] < 1000]
train_data = train_data[train_data["item_cnt_day"] < 1000]


# format date column to datetime
train_data["date"] = pd.to_datetime(train_data["date"])
train_data["month_and_year"] = train_data["date"].dt.strftime("%Y-%m")

# filter out item/shop combinations where that have no sales in the last 3 months
last_three_months = sorted(train_data["month_and_year"].unique())[-3:]
last_three_months = train_data[train_data["month_and_year"].isin(last_three_months)]
last_three_months_sales = last_three_months.groupby(["month_and_year", "shop_id", "item_id"])["item_cnt_day"].sum().reset_index()
combinations_with_sales = last_three_months_sales[last_three_months_sales["item_cnt_day"] > 0][["shop_id", "item_id"]]

# get list of all months we have data for
first_month = train_data["date"].min().strftime("%Y-%m")
last_month = train_data["date"].max().strftime("%Y-%m")
all_months = train_data["month_and_year"].unique()


# build df with sales per month
matrix = list(
    product(
        all_months,
        combinations_with_sales.values
    )
)
train_df = pd.DataFrame(matrix, columns = ["month_and_year", "arr"])
train_df["shop_id"] = train_df["arr"].str[0]
train_df["item_id"] = train_df["arr"].str[1]
train_df.drop(columns = ["arr"], inplace = True)

# merge datapoints to df
train_df = train_df.merge(item_data[["item_id", "item_category_id"]], how = "left", on = "item_id")
train_df = train_df.merge(category_names, how = "left", on = "item_category_id")
train_df = train_df.merge(item_names[["item_id", "item_name"]], how = "left", on = "item_id")
train_df = train_df.merge(shop_names[["shop_name", "shop_id"]], how = "left", on = "shop_id")

prediction_data = prediction_data.merge(item_data[["item_id", "item_category_id"]], how = "left", on = "item_id")
prediction_data = prediction_data.merge(category_names, how = "left", on = "item_category_id")
prediction_data = prediction_data.merge(item_names[["item_id", "item_name"]], how = "left", on = "item_id")
prediction_data = prediction_data.merge(shop_names[["shop_name", "shop_id"]], how = "left", on = "shop_id")


# add broad category to df
train_df[['broad_category', 'subcategory']] = train_df['item_category_name'].str.split('-', 1, expand=True)
train_df["broad_category"] = train_df["broad_category"].str.strip().str.lower()
train_df["subcategory"] = train_df["subcategory"].str.strip().str.lower()
train_df.drop(columns = ["item_category_name"], inplace = True)

prediction_data[['broad_category', 'subcategory']] = prediction_data['item_category_name'].str.split('-', 1, expand=True)
prediction_data["broad_category"] = prediction_data["broad_category"].str.strip().str.lower()
prediction_data["subcategory"] = prediction_data["subcategory"].str.strip().str.lower()
prediction_data.drop(columns = ["item_category_name"], inplace = True)



# add city
train_df[['city', 'shop_name']] = train_df['shop_name'].str.split(' ', 1, expand=True)
train_df["city"] = train_df["city"].str.strip().str.lower()

prediction_data[['city', 'shop_name']] = prediction_data['shop_name'].str.split(' ', 1, expand=True)
prediction_data["city"] = prediction_data["city"].str.strip().str.lower()



# merge item sales to df
item_sales = train_data.groupby(["month_and_year", "shop_id", "item_id"])["item_cnt_day"].sum().reset_index().rename(columns = {"item_cnt_day": "item_cnt_month"})
train_df = train_df.merge(item_sales, how = "left", on = ["month_and_year", "shop_id", "item_id"])
train_df["item_cnt_month"].fillna(0, inplace = True)


train_df.drop(columns = ["item_category_id", "item_name", "subcategory"], inplace = True)
prediction_data.drop(columns = ["item_category_id", "item_name", "subcategory"], inplace = True)


# add shop type
train_df["shop_type"] = train_df["shop_name"].str.extract(r"(^[A-Z]+[a-z]* )")
train_df.loc[train_df["shop_name"].str.startswith("shopping center"), "shop_type"] = "shopping center"
train_df.loc[train_df["shop_name"].str.startswith("warehouse"), "shop_type"] = "warehouse"
train_df.loc[train_df["shop_name"].str.startswith("online store"), "shop_type"] = "online store"
train_df.loc[train_df["shop_name"].str.startswith("shopping and entertainment complex"), "shop_type"] = "shopping and entertainment complex"
train_df["shop_type"].fillna("other", inplace = True)
train_df["shop_type"] = train_df["shop_type"].str.lower().str.strip()

prediction_data["shop_type"] = prediction_data["shop_name"].str.extract(r"(^[A-Z]+[a-z]* )")
prediction_data.loc[prediction_data["shop_name"].str.startswith("shopping center"), "shop_type"] = "shopping center"
prediction_data.loc[prediction_data["shop_name"].str.startswith("warehouse"), "shop_type"] = "warehouse"
prediction_data.loc[prediction_data["shop_name"].str.startswith("online store"), "shop_type"] = "online store"
prediction_data.loc[prediction_data["shop_name"].str.startswith("shopping and entertainment complex"), "shop_type"] = "shopping and entertainment complex"
prediction_data["shop_type"].fillna("other", inplace = True)
prediction_data["shop_type"] = prediction_data["shop_type"].str.lower().str.strip()


# format broad category
train_df.loc[train_df["broad_category"].str.contains("games"), "broad_category"] = "games"
prediction_data.loc[prediction_data["broad_category"].str.contains("games"), "broad_category"] = "games"


# merge dateblocknumber to df
all_months = sorted(train_df["month_and_year"].unique())
date_block_num = list(enumerate(all_months))
date_block_num = pd.DataFrame(date_block_num, columns = ["date_block_num", "month_and_year"])
train_df = train_df.merge(date_block_num, how = "left", on = "month_and_year")



# format prediction data month
last_month = train_df["month_and_year"].max()
last_year, last_month = last_month.split("-")
prediction_month = (int(last_month) + 1) % 12
prediction_year = int(last_year) if prediction_month != 1 else int(last_year) + 1
prediction_month = f"{prediction_year}-{prediction_month}"
prediction_data["month_and_year"] = prediction_month

# merge month name to df
month_name_df = pd.DataFrame(
    all_months + [prediction_month],
    columns = ["month_and_year"]
)
month_name_df["datetime"] = pd.to_datetime(month_name_df["month_and_year"])
month_name_df["month_name"] = month_name_df["datetime"].dt.month_name()

train_df = train_df.merge(month_name_df[["month_and_year", "month_name"]], how = "left", on = "month_and_year")
prediction_data = prediction_data.merge(month_name_df[["month_and_year", "month_name"]], how = "left", on = "month_and_year")

# merge number of days in that month to df
days_per_month = pd.DataFrame(
    zip(all_months + [prediction_month],
    list(map(lambda x: pd.Period(x).days_in_month, all_months + [prediction_month]))),
    columns = ["month_and_year", "days_in_month"]
)
train_df = train_df.merge(days_per_month, how = "left", on = "month_and_year")
prediction_data = prediction_data.merge(days_per_month, how = "left", on = "month_and_year")


# add month lag to data

month_lag_count = 3
train_with_lag = train_df[train_df["date_block_num"] >= month_lag_count].copy(deep = True)
dataframe_copy = train_df[["date_block_num", "shop_id", "item_id", "item_cnt_month"]].copy(deep = True)
prediction_data["date_block_num"] = train_df["date_block_num"].max() + 1

for i in tqdm(range(month_lag_count), desc = "Adding month lag..."):
    train_with_lag["date_block_num"] -= 1
    prediction_data["date_block_num"] -= 1
    
    train_with_lag = train_with_lag.merge(dataframe_copy.rename(columns = {"item_cnt_month": f"lag_count_-{i+1}"}), how = "left", on = ["date_block_num", "shop_id", "item_id"])
    prediction_data = prediction_data.merge(dataframe_copy.rename(columns = {"item_cnt_month": f"lag_count_-{i+1}"}), how = "left", on = ["date_block_num", "shop_id", "item_id"])

train_df = train_with_lag

prediction_data.set_index("ID", inplace = True)


train_df.drop(columns = ["month_and_year", "shop_name"], inplace = True)
prediction_data.drop(columns = ["month_and_year", "shop_name"], inplace = True)

prediction_data = prediction_data.reindex(sorted(prediction_data.columns), axis=1)
train_df = train_df.reindex(sorted(train_df.columns), axis=1)

train_df = pd.get_dummies(
    train_df
)

prediction_data = pd.get_dummies(
    prediction_data
)


for col in train_df.columns:
    if col not in prediction_data.columns:
        prediction_data[col] = 0

prediction_data.drop(columns = ["broad_category_pc", "item_cnt_month"], inplace=True)

train_df = train_df.reindex(sorted(train_df.columns), axis = 1)
prediction_data = prediction_data.reindex(sorted(prediction_data.columns), axis = 1)

assert len(train_df.columns) == len(prediction_data.columns) + 1, "Error: Wrong number of Features"

train_df.to_csv("format/train_data.csv", index = False)
prediction_data.to_csv("format/prediction_data.csv", index = False)


