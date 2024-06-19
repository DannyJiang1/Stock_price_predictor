from typing import List


def print_stats(percent_changes: List[float], stock: str, start_date: str):

    print(percent_changes)

    gains = 0
    ng = 0
    losses = 0
    nl = 0
    totalR = 1

    for i in percent_changes:
        if i > 0:
            gains += i
            ng += 1
        else:
            losses += i
            nl += 1
        totalR = totalR * ((i / 100) + 1)

    totalR = round((totalR - 1) * 100, 2)

    if ng > 0:
        avgGain = gains / ng
        maxR = str(max(percent_changes))
    else:
        avgGain = 0
        maxR = "undefined"

    if nl > 0:
        avgLoss = losses / nl
        maxL = str(min(percent_changes))
        ratio = str(-avgGain / avgLoss)
    else:
        avgLoss = 0
        maxL = "undefined"
        ratio = "inf"

    if ng > 0 or nl > 0:
        battingAvg = ng / (ng + nl)
    else:
        battingAvg = 0

    print()
    print(
        "Results for "
        + stock
        + " going back to "
        + start_date
        + ", Sample size: "
        + str(ng + nl)
        + " trades"
    )
    print("Batting Avg: " + str(battingAvg))
    print("Gain/loss ratio: " + ratio)
    print("Average Gain: " + str(avgGain))
    print("Average Loss: " + str(avgLoss))
    print("Max Return: " + maxR)
    print("Max Loss: " + maxL)
    print(
        "Total return over " + str(ng + nl) + " trades: " + str(totalR) + "%"
    )
