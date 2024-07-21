import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from utils import print_stats


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
