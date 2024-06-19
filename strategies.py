import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from utils import print_stats

# Future Indicators:
# RSI
# MACD
# BOLLINGER BAND


""" indicators: {
    rsi: {
        period: int,
        average_type: str,
    },
    bollinger_bands: {
        period: int,
        num_std_dev : int,
    },
    macd: {
        fast_period: int,
        slow_period: int,
        signal_period: int
    }
    emas: List[int]
}"""


def add_technical_indicators(df, indicators: dict):
    # RSI

    if "rsi" in indicators:
        period = indicators["rsi"]["period"]
        average_type = indicators["rsi"]["average_type"]
        rsi(df, period, average_type)

    # Bollinger Bands
    if "bollinger_bands" in indicators:
        period = indicators["bollinger_bands"]["period"]
        num_std_dev = indicators["bollinger_bands"]["num_std_dev"]
        bollinger_bands(df, period, num_std_dev)

    # MACD
    if "macd" in indicators:
        fast_period = indicators["macd"]["fast_period"]
        slow_period = indicators["macd"]["slow_period"]
        signal_period = indicators["macd"]["signal_period"]
        macd(df, fast_period, slow_period, signal_period)

        # EMAs

    if "emas" in indicators:
        for period in indicators["emas"]:
            df[f"EMA_{period}"] = (
                df["Close"].ewm(span=period, adjust=False).mean()
            )
    return df


def rsi(df, period=14, average_type="ema"):
    # Calculate the difference in price from the previous step
    delta = df["Close"].diff()

    # Make two series: one for gains and one for losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Calculate the exponential or simple moving average
    if average_type.lower() == "ema":
        average_gain = gain.ewm(com=(period - 1), min_periods=period).mean()
        average_loss = loss.ewm(com=(period - 1), min_periods=period).mean()
    elif average_type.lower() == "sma":
        average_gain = gain.rolling(window=period, min_periods=period).mean()
        average_loss = loss.rolling(window=period, min_periods=period).mean()

    # Calculate the RS
    RS = average_gain / average_loss

    # Calculate the RSI and add it to the DataFrame
    df["RSI"] = 100 - (100 / (1 + RS))


def bollinger_bands(df, period=20, num_std_dev=2):
    # Calculate the moving average (Middle Band)
    df["Middle Band"] = df["Close"].rolling(window=period).mean()

    # Calculate the standard deviation
    std_dev = df["Close"].rolling(window=period).std()

    # Calculate the Upper and Lower Bollinger Bands
    df["Upper Band"] = df["Middle Band"] + (std_dev * num_std_dev)
    df["Lower Band"] = df["Middle Band"] - (std_dev * num_std_dev)


def macd(df, fast_period=12, slow_period=26, signal_period=9):
    # Calculate the fast and slow exponential moving averages
    df["EMA_Fast"] = df["Close"].ewm(span=fast_period, adjust=False).mean()
    df["EMA_Slow"] = df["Close"].ewm(span=slow_period, adjust=False).mean()

    # Calculate the MACD line and Signal line
    df["MACD"] = df["EMA_Fast"] - df["EMA_Slow"]
    df["Signal Line"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()


def emas(df, emas: List[int]):
    for ema in emas:
        df["Ema_" + str(ema)] = df["Close"].ewm(span=ema, adjust=False).mean()


def st_lt_ema(
    stock: str, df, short_term_emas: List[int], long_term_emas: List[int]
):
    for ema in short_term_emas:
        df["Ema_" + str(ema)] = round(
            df.iloc[:, 4].ewm(span=ema, adjust=False).mean(), 2
        )

    for ema in long_term_emas:
        df["Ema_" + str(ema)] = round(
            df.iloc[:, 4].ewm(span=ema, adjust=False).mean(), 2
        )

    entered = False
    percent_changes = []
    ema_values = {}
    days = []

    for ema in short_term_emas:
        ema_values[f"Ema_{ema}"] = []

    for ema in long_term_emas:
        ema_values[f"Ema_{ema}"] = []

    for day in df.index:
        days.append(day)
        for ema in short_term_emas:
            ema_values[f"Ema_{ema}"].append(df[f"Ema_{ema}"][day])
        for ema in long_term_emas:
            ema_values[f"Ema_{ema}"].append(df[f"Ema_{ema}"][day])

    DF = pd.DataFrame(ema_values, index=days)

    plt.figure(figsize=(10, 6))
    colors = ["blue"] * len(short_term_emas) + ["red"] * len(long_term_emas)
    for i, column in enumerate(DF.columns):
        plt.plot(DF.index, DF[column], label=column, color=colors[i])
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.title("EMA Values Over Time")
    plt.xlabel("Date")
    plt.ylabel("EMA Value")

    for i, day in enumerate(df.index):
        min_short_term = min([df[f"Ema_{e}"][day] for e in short_term_emas])
        max_long_term = max([df[f"Ema_{e}"][day] for e in long_term_emas])
        # print(day)

        daily_adj_close = df["Adj Close"][i]

        if min_short_term > max_long_term:
            if not entered:
                entry_price = daily_adj_close
                entered = True
                plt.axvline(
                    x=day, color="green", linestyle="--", label="Vertical Line"
                )
                print(f"Buying at {daily_adj_close} on {day}")
        else:
            if entered:
                exit_price = daily_adj_close
                entered = False
                plt.axvline(
                    x=day,
                    color="yellow",
                    linestyle="--",
                    label="Vertical Line",
                )
                print(f"Selling at {daily_adj_close} on {day}")
                percent_changes.append(((exit_price / entry_price) - 1) * 100)

        if i == df["Adj Close"].count() - 1 and entered:
            exit_price = daily_adj_close
            entered = False
            plt.axvline(
                x=day, color="yellow", linestyle="--", label="Vertical Line"
            )
            print(f"Selling at {daily_adj_close} on {day}")
            percent_changes.append(((exit_price / entry_price) - 1) * 100)

    print_stats(
        percent_changes=percent_changes,
        stock=stock,
        start_date=str(df.index[0]),
    )

    plt.savefig("ema_plot.png")
    plt.close()
