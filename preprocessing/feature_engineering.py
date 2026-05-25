import pandas as pd
def feature(train):
    train["flow_lag_1"] = (train["streamflow_today_cumecs"].shift(1))
    train["flow_lag_3"] = (train["streamflow_today_cumecs"].shift(3))
    train["flow_lag_7"] = (train["streamflow_today_cumecs"].shift(7))
    train["lag_14"] = (train["streamflow_today_cumecs"].shift(14))
    train["lag_30"] = (train["streamflow_today_cumecs"].shift(30))
    train["rolling_mean_7"] = (train["streamflow_today_cumecs"].rolling(7).mean())
    train["rolling_std_7"] = (train["streamflow_today_cumecs"].rolling(7).std())
    train["diff_1"] = (train["streamflow_today_cumecs"] -train["flow_lag_1"])
    train["ratio_7"] = (train["streamflow_today_cumecs"] /(train["rolling_mean_7"] + 1e-5))
    train["ema_7"] = (train["streamflow_today_cumecs"].ewm(span=7).mean())
    train["rolling_max_7"] = (train["streamflow_today_cumecs"].rolling(7).max())
    remove_cols = [
        "month",
        "antecedent_rain_30d_sum",
        "antecedent_rain_60d",
        "rolling_max_7",
        "rain_basinsize_interaction",
        "uparea_rain_interaction",
        "ratio_7",
    ]

    return train.drop(remove_cols,axis=1)