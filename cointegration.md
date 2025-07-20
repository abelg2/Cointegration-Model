

# Trade-Ideas-Projects
# Cointegration Model Verion 1
## Description/Additional Info
Using an equity screener, I narrowed down the extensive list to largely tech-based large-cap stocks and added a few more constraints to a small subset as shown below:
##

```Python
import pandas as pd
import yfinance as yf
import itertools
from datetime import datetime as dt
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

start_str = "2023-01-01"
end_str = dt.today()-pd.Timedelta(days=2)


ticker_list = ["CDNS", "DDOG", "MSTR", "PAYX", "SHOP", "WDAY", "SNOW", "PTC"]


def pricing(ticker):
    try:
        data = yf.download(ticker,
                           start = start_str,
                           end = end_str,
                           auto_adjust = False,
                           progress = False
                           )
        if not data.empty:
            return data["Close"]
    except Exception as e:
        print(f"Error: {e}")
    return None

price_df = pd.DataFrame()

for ticker in ticker_list:
    series = pricing(ticker)
    if series is not None:
        price_df[ticker] = series
price_df.dropna(inplace=True)
price_df.reset_index(inplace=True)

# This assumes a 30-month lookback -- it may be worth lengthening or shortening duration

#def cointegration_test(stock1, stock2):
    #score, p_value, _ = coint(price_df[stock1], price_df[stock2])
    #return round(score, 4), round(p_value, 4)

for stock1, stock2 in itertools.combinations(ticker_list, 2):
    if stock1 in price_df.columns and stock2 in price_df.columns:
        

        score, p_value = cointegration_test(stock1, stock2)
        print(f"{stock1} & {stock2}:")
        print(f" Cointegration ScoreL {score}")
        print(f"P_value: {p_value}")

# Top cointegrated pairs (p<0.05):
## 
DDOG & PTC [score: -4.21, p:0.0035]
CDNS & PTC [score: -4.136, p:0.0045]
MSTR & PAYX [score: -3.986, p:0.0075]

Interestingly, DDOG, PTC, & CDNS appear to be cointegrated. It may be worth exploring a 3 sided relationship with a Johansen Cointegration Tes
For this run I chose the first pair to run my tests
##

ticker1 = "PTC"
ticker2 = "DDOG"
X = sm.add_constant(price_df[ticker1])
model = sm.OLS(price_df[ticker2], X).fit()
hedge_ratio = model.params[ticker1]
print(f"Hedge_ratio: {hedge_ratio:.4f}") # 0.9392

window = 25
price_df["Hedge_Spread"] = price_df[ticker1] - hedge_ratio * price_df[ticker2]

price_df["HS_mean"] = price_df["Hedge_Spread"].rolling(window).mean()
price_df["HS_STD"] = price_df["Hedge_Spread"].rolling(window).std()
price_df["Zscore"] = (price_df["Hedge_Spread"] - price_df["HS_mean"])/price_df["HS_STD"]
price_df.dropna(inplace=True)


plt.plot(price_df["Date"], price_df["Zscore"], label="Z-Score")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.grid(True)
#plt.show()

# upper/lower - 1.98
# exit - 0.25

class PairsTradeSimulator():
    def __init__(self, df, ticker1, ticker2, capital=100000, trading_threshold = 1.95, exiting_threshold=0.25):

        self.df = df.copy()
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.capital = capital
        self.trading_threshold = trading_threshold
        self.exiting_threshold = exiting_threshold
        self.position = None
        self.cash = capital
        self.positions = []
        self.trades = []
        self.portfolio_value = []

    def run(self):

        for _, row in self.df.iterrows():
            z = row["Zscore"]
            price1 = row[self.ticker1]
            price2 = row[self.ticker2]

            if self.position is None:
                if z > self.trading_threshold:
                    self.enter_short(price1, price2, row["Date"])
                elif z < -self.trading_threshold:
                    self.enter_long(price1, price2, row["Date"])
            elif self.position in ["long", "short"]:
                if abs(z) < self.exiting_threshold:
                    self.exit_position(price1, price2, row["Date"])

            if self.position == "long":
                pos_value = row[self.ticker1] * self.entry["Qty1"]-row[self.ticker2] * self.entry["Qty2"]
            elif self.position == "short":
                pos_value = -row[self.ticker1] * self.entry["Qty1"]+row[self.ticker2] * self.entry["Qty2"]
            else:
                pos_value = 0
            self.portfolio_value.append(self.cash + pos_value)


    def enter_long(self, price1, price2, date):

        qty1 = (self.capital / 2) / price1
        qty2 = (self.capital / 2) / price2
        self.position = "long"
        self.entry = {"Price1": price1, "Price2": price2,
                   "Qty1": qty1, "Qty2": qty2, "Date": date}
        print(f"[{date}] ENTER LONG {self.ticker1} @ {price1:.2f}, SHORT {self.ticker2} @ {price2:.2f}")

    def enter_short(self, price1, price2, date):

        qty1 = (self.capital / 2) / price1
        qty2 = (self.capital / 2) / price2
        self.position = "short"
        self.entry = {"Price1": price1, "Price2": price2,
                   "Qty1": qty1, "Qty2": qty2, "Date": date}
        print(f"[{date}] ENTER SHORT {self.ticker1} @ {price1:.2f}, LONG {self.ticker2} @ {price2:.2f}")

    def exit_position(self, price1, price2, date):

        e = self.entry
        if self.position == "long":
            pnl = (price1 - e["Price1"]) * e["Qty1"] + (e["Price2"] - price2) * e["Qty2"]
        elif self.position == "short":
            pnl = (e["Price1"] - price1) * e["Qty1"] + (price2 - e["Price2"]) * e["Qty2"]
        self.cash += pnl
        print(f"[{date}] EXIT {self.position} | P&L {pnl:.3f} | Cash: {self.cash:.2f}")
        self.trades.append({"Entry Date": e["Date"], "Exit Date": date, "PNL": pnl})
        self.position = None
        self.entry = None

    def summary(self):
        print("______________Trade_Summary_______________")
        for t in self.trades:
            print(f"{t['Entry Date']} -> {t['Exit Date']}: P&L = ${t['PNL']}")
        print(f"Final Cash: {self.cash}")


sim = PairsTradeSimulator(price_df, ticker1, ticker2)
sim.run()
sim.summary()
```
# Starting portfolio value: $100K
# Ending portfolio value: $150k
# In future iterations, I will consider a more dynamic hedge_ratio. Also, what are the optimal window ranges for rolling averages and standard deviations? What about trading_thresholds (enter, exit, etc)

    
