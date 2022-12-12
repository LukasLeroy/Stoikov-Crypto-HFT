import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

"""
Author: Jimmy Yeung
Date: 20/03/2022
"""


# np.random.seed(1)


class StoikovModel:
    def __init__(self):
        pass

    def simulate_bm(self, s0=100, sigma=2, T=1, dt=1 / 3600, M=3600):
        """
        Simulates a Brownian motion path. We assume that volatility remains constant.

        :param s0: int
            Initial value of Brownian motion
        :param sigma: int
            Volatility input as a percent (e.g. 2 = 2%)
        :param T: int
            Expiry time
        :param dt: float
            Time increments
        :param M: int
            Number of steps = T/dt
        :return: list
            Simulated Brownian motion as a list
        """
        sigma = s0 * sigma / 100  # convert sigma from percent
        s = [0 for _ in range(M + 1)]
        s[0] = s0
        for t in range(1, M + 1):
            s[t] = s[t - 1] + sigma * math.sqrt(dt) * np.random.normal()
        return s

    def convert_data(self, coin):
        """
        Converts the market data provided by Kraken from unixtime and calculates dt

        :param coin: str
            Cryto tickers e.g., ETH, XBT, SOL
        :return: None
        """
        df = pd.read_csv(f'Q4 2021/{coin}/{coin}USD.csv')
        df.columns = ['Datetime', 'Mid', 'Size']
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
        df['dt'] = (df['Datetime'] - df['Datetime'].shift(1)).dt.total_seconds().shift(-1)
        df.set_index('Datetime', inplace=True)
        df.to_csv(f'data/{coin}USD.csv')

    def read_data(self, pair):
        """
        Reads converted market data csv file.
        """
        df = pd.read_csv(f'data/{pair}.csv', index_col=0)
        return df

    def _backtest(self, pair, s0, bm_sigma, T, dt, M, q0, gamma, k, A, plot):
        """
        Computes the trading logic and backtests the market making trading strategy.

        :param pair: str
            e.g., "XBTUSD", "ETHUSD" or "bm" for simulated Brownian motion.
        :param s0: int
            Initial value of Brownian motion or market data.
        :param bm_sigma: float
            Volatility value for Brownian motion. Not used for real market data.
        :param T: int
            End time.
        :param dt: float
            Time increments.
        :param M: int
            No. of iterations (=T/dt).
        :param q0: int
            Initial inventory position.
        :param gamma: float
            Risk aversion parameter.
        :param k: float
            Trading parameter. Larger k = smaller spread.
        :param A: int
            Trading parameter. Represents the frequency of market orders.
        :param plot: bool
            Whether to plot graphs.
        :return: float
            Return final PnL at the end of backtest.
        """
        inventory_limit = 20
        maker_fee = 0.0001  # 1bps
        vol_calc_lookback = 50  # lookback period to calculate volatility
        quote_size = 1  # Trading strategy quote size

        if pair == 'bm':  # simulate brownian motion
            s = self.simulate_bm(s0, bm_sigma, T, dt, M)
        else:  # read market data
            df = self.read_data(pair)
            dts = list(df['dt'][:M])
            s = list(df['Mid'])[:M]

        # initialise
        bids1 = [None for _ in range(M)]
        bids2 = [None for _ in range(M)]
        asks1 = [None for _ in range(M)]
        asks2 = [None for _ in range(M)]
        theo = [None for _ in range(M)]
        spreads = [None for _ in range(M)]
        delta_bid1 = [0 for _ in range(M)]
        delta_bid2 = [0 for _ in range(M)]
        delta_ask1 = [0 for _ in range(M)]
        delta_ask2 = [0 for _ in range(M)]
        inventory = [0 for _ in range(M)]
        cash = [0 for _ in range(M)]
        equity = [0 for _ in range(M)]
        sigmas = [None for _ in range(M)]

        theo[0] = None
        bids1[0] = None
        bids2[0] = None
        asks1[0] = None
        asks2[0] = None
        delta_bid1[0] = 0
        delta_bid2[0] = 0
        delta_ask1[0] = 0
        delta_ask2[0] = 0
        inventory[0] = q0
        cash[0] = 0
        equity[0] = 0

        for t in range(vol_calc_lookback, M):

            # Calculate volatility
            past50_mids_df = pd.DataFrame(s[t - vol_calc_lookback:t], columns=['mids'])
            past50_mids_df['Log Return'] = (np.log(past50_mids_df['mids']) - np.log(past50_mids_df['mids'].shift(1)))
            sigma = np.sqrt(M) * past50_mids_df['Log Return'].std() * 100
            sigmas[t] = sigma

            # Theoretical price and spread calculation
            theo[t] = s[t] - (inventory[t - 1] * gamma * (sigma ** 2) * (T - t * dt)) * s[t] / 100
            spreads[t] = (gamma * (sigma ** 2) * (T - t * dt) + (2 / gamma) * math.log(1 + (gamma / k))) * s[t] / 100

            # Calculate level 1 and level 2 bid and asks
            bids1[t] = theo[t] - spreads[t] / 2
            bids2[t] = theo[t] - spreads[t]
            asks1[t] = theo[t] + spreads[t] / 2
            asks2[t] = theo[t] + spreads[t]

            # Calculate percentage difference of bids and asks from mid
            delta_bid1[t] = (s[t] - bids1[t]) / s[t]
            delta_bid2[t] = (s[t] - bids2[t]) / s[t]
            delta_ask1[t] = (asks1[t] - s[t]) / s[t]
            delta_ask2[t] = (asks2[t] - s[t]) / s[t]

            # Poisson intensity for probabilities
            lambda_ask1 = A * np.exp(-k * delta_ask1[t])
            lambda_ask2 = A * np.exp(-k * delta_ask2[t])
            lambda_bid1 = A * np.exp(-k * delta_bid1[t])
            lambda_bid2 = A * np.exp(-k * delta_bid2[t])

            # Calculate trading probabilities
            if pair != 'bm':  # if using market data we take into account the time difference between increments
                prob_ask1 = lambda_ask1 * dts[t] / 3600
                prob_ask2 = lambda_ask2 * dts[t] / 3600
                prob_bid1 = lambda_bid1 * dts[t] / 3600
                prob_bid2 = lambda_bid2 * dts[t] / 3600
            else:  # else we just use a one second time increment of 1/3600
                prob_ask1 = lambda_ask1 * dt
                prob_ask2 = lambda_ask2 * dt
                prob_bid1 = lambda_bid1 * dt
                prob_bid2 = lambda_bid2 * dt

            # Random sample to simulate trades
            uniform_ask = random.random()
            uniform_bid = random.random()

            # Trading logic
            if prob_bid1 > uniform_bid and prob_ask1 < uniform_ask:  # market order hit level 1 bid but not the ask
                if inventory[t - 1] >= inventory_limit:  # check inventory limit - cannot trade if exceeds
                    inventory[t] = inventory[t - 1]
                    cash[t] = cash[t - 1]
                else:
                    inventory[t] = inventory[t - 1] + quote_size
                    cash[t] = cash[t - 1] - bids1[t] - bids1[t] * maker_fee
                    if prob_bid2 > uniform_bid:  # if level 2 bid hit
                        inventory[t] = inventory[t] + quote_size
                        cash[t] = cash[t] - bids2[t] - bids2[t] * maker_fee

            if prob_bid1 < uniform_bid and prob_ask1 > uniform_ask:  # market order lifting level 1 ask but not the bid
                if inventory[t - 1] <= -inventory_limit:  # check inventory
                    inventory[t] = inventory[t - 1]
                    cash[t] = cash[t - 1]
                else:
                    inventory[t] = inventory[t - 1] - quote_size
                    cash[t] = cash[t - 1] + asks1[t] - asks1[t] * maker_fee
                    if prob_ask2 > uniform_ask:  # if level 2 ask lifted
                        inventory[t] = inventory[t] - quote_size
                        cash[t] = cash[t] + asks2[t] - asks2[t] * maker_fee

            if prob_bid1 < uniform_bid and prob_ask1 < uniform_ask:  # neither hit or lifted
                inventory[t] = inventory[t - 1]
                cash[t] = cash[t - 1]

            if prob_bid1 > uniform_bid and prob_ask1 > uniform_ask:  # both bid and ask hit and lifted
                if inventory[t - 1] == inventory_limit:  # can't buy, only sell
                    inventory[t] = inventory[t - 1] - quote_size
                    cash[t] = cash[t - 1] + asks1[t] - asks1[t] * maker_fee
                elif inventory[t - 1] == -inventory_limit:  # can't sell, only buy
                    inventory[t] = inventory[t - 1] + quote_size
                    cash[t] = cash[t - 1] - bids1[t] - bids1[t] * maker_fee
                else:  # can buy and sell
                    inventory[t] = inventory[t - 1]
                    cash[t] = cash[t - 1] - bids1[t] - bids1[t] * maker_fee
                    cash[t] = cash[t] + asks1[t] - asks1[t] * maker_fee
                    if prob_bid2 > uniform_bid and prob_ask2 > uniform_ask:  # if level 2 bid and ask hit and lifted
                        inventory[t] = inventory[t - 1]
                        cash[t] = cash[t] - bids2[t] - bids2[t] * maker_fee
                        cash[t] = cash[t] + asks2[t] - asks2[t] * maker_fee

            equity[t] = cash[t] + inventory[t] * s[t]  # update equity

        # Initialise dataframe for plots
        prices_df = pd.DataFrame([asks2, asks1, s, bids1, bids2])
        prices_df = prices_df.T
        prices_df.columns = ['Ask2', 'Ask1', 'Mid', 'Bid1', 'Bid2']

        spreads_df = pd.DataFrame(spreads, columns=['Spread'])
        sigmas_df = pd.DataFrame(sigmas, columns=['Volatility'])

        equity_df = pd.DataFrame(equity, columns=['Equity'])
        positions_df = pd.DataFrame(inventory, columns=['Inventory'])
        cash_df = pd.DataFrame(cash, columns=['Cash'])

        if plot:
            fig = plt.figure(figsize=(25, 15))
            ax_dict = fig.subplot_mosaic(
                [
                    ["price", "price", 'price', 'price', 'price'],
                    ["spread", "vol", 'inv', 'cash', 'equity'],
                ],
                gridspec_kw={
                    "height_ratios": [3, 1],
                },
            )
            fig.suptitle(f"Market Making Strategy on {pair}", fontsize=28)
            prices_df.plot(ax=ax_dict['price'], grid=True, xlabel='t', ylabel='$', fontsize=20,
                           legend=True,
                           color=['#D10A0A', '#D10A0A', '#0A39D1', '#0A9C28', '#0A9C28'],
                           style=['--', '-', '-', '-', '--'])
            ax_dict['price'].legend(fontsize=20)
            spreads_df.plot(ax=ax_dict['spread'], title='Spread', grid=True, xlabel='t', legend=False)
            ax_dict['spread'].title.set_size(20)
            sigmas_df.plot(ax=ax_dict['vol'], title='50-tick Rolling Volatility (%)', grid=True, xlabel='t',
                           legend=False)
            ax_dict['vol'].title.set_size(20)
            positions_df.plot(ax=ax_dict['inv'], title='Inventory', grid=True, xlabel='t', legend=False)
            ax_dict['inv'].title.set_size(20)
            cash_df.plot(ax=ax_dict['cash'], title='Cash', grid=True, xlabel='t', ylabel='$', legend=False)
            ax_dict['cash'].title.set_size(20)
            equity_df.plot(ax=ax_dict['equity'], title='Equity', grid=True, xlabel='t', ylabel='$', legend=False)
            ax_dict['equity'].title.set_size(20)
            plt.ticklabel_format(useOffset=False)
            plt.tight_layout()
            plt.show()

        return equity[-1]  # return final equity (PnL)

    def backtest_bm(self, s0=100, bm_sigma=2, T=1, dt=1 / 3600, M=3600, q0=0, gamma=0.1, k=1.5, A=140, plot=False):
        """
        Backtests trading strategy for simulated Brownian motion. Input parameters same as _backtest.

        :return: float
            Final equity (PnL) of simulated Brownian motion trading strategiy.
        """
        return self._backtest('bm', s0=s0, bm_sigma=bm_sigma, T=T, dt=dt, M=M, q0=q0, gamma=gamma, k=k, A=A, plot=plot)

    def backtest_data(self, pair='XBTUSD', T=1, dt=1 / 3600, M=3600, q0=0, gamma=0.1, k=10, A=140, plot=False):
        """
        Backtests trading strategy for coin pair. Input parameters same as _backtest.

        :return: float
            Final equity (PnL) of simulated Brownian motion trading strategiy.
        """
        df = self.read_data(pair)
        s = list(df['Mid'])[:M]
        s0 = s[0]
        return self._backtest(pair, s0=s0, bm_sigma=None, T=T, dt=dt, M=M, q0=q0, gamma=gamma, k=k, A=A, plot=plot)

    def _simulate_many_bm(self, s0=100, bm_sigma=2, T=1, dt=1 / 3600, q0=0, gamma=0.1, k=1.5, A=140, sim_num=500):
        """
        :param sim_num: int
            No. of times to backtest trading strategy on Brownian motion.
        :return: List
            Returns a list of the final equity (PnL) values for the simulations.
        """
        final_equities = []
        for i in range(sim_num):
            print('sigma', bm_sigma, 'gamma', gamma, 'num', i)
            final_equities.append(
                self.backtest_bm(s0=s0, bm_sigma=bm_sigma, T=T, dt=dt, q0=q0, gamma=gamma, k=k, A=A, plot=False))
        return final_equities

    def plot_bm_pnl_sigma(self, s0=100, bm_sigmas=[1, 2, 5, 10], T=1, dt=1 / 3600, q0=0, gamma=0.1, k=1.5, A=140,
                          sim_num=100):
        """
        Plot final equity PnL histogram for backtested Brownian motions with different sigma values.

        :param bm_sigmas: List
            List of volatilities for Brownian motion simulations
        """
        final_equities_df = pd.DataFrame()
        for bm_sigma in bm_sigmas:
            final_equities = self._simulate_many_bm(s0=s0, bm_sigma=bm_sigma, T=T, dt=dt, q0=q0, gamma=gamma, k=k, A=A,
                                                    sim_num=sim_num)
            final_equities_df[str(bm_sigma)] = final_equities

        fig, axes = plt.subplots(figsize=(15, 10))
        for bm_sigma in bm_sigmas:
            final_equities_df[str(bm_sigma)].hist(ax=axes, bins=30, label=f'$\sigma={bm_sigma}$', alpha=0.2)
        plt.title(f'Histogram of equity at end of {sim_num} simulations', fontsize=24)
        plt.xlabel('Final Equity ($)', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.legend(fontsize=16)
        plt.show()

    def plot_bm_pnl_gamma(self, s0=100, bm_sigma=2, T=1, dt=1 / 3600, q0=0, gammas=[0.5, 0.1, 0.01], k=1.5, A=140,
                          sim_num=100):
        """
        Plot final equity PnL histogram for backtested Brownian motions with different gamma values.

        :param gammas: List
            List of risk-aversion parameters for Brownian motion simulations
        """
        final_equities_df = pd.DataFrame()
        for gamma in gammas:
            final_equities = self._simulate_many_bm(s0=s0, bm_sigma=bm_sigma, T=T, dt=dt, q0=q0, gamma=gamma, k=k, A=A,
                                                    sim_num=sim_num)
            final_equities_df[str(gamma)] = final_equities
        fig, axes = plt.subplots(figsize=(15, 10))
        for gamma in gammas:
            final_equities_df[str(gamma)].hist(ax=axes, bins=30, label=f'$\gamma={gamma}$', alpha=0.2)
        plt.title(f'Histogram of equity at end of {sim_num} simulations', fontsize=24)
        plt.xlabel('Final Equity ($)', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.legend(fontsize=16)
        plt.show()


if __name__ == '__main__':
    bot = StoikovModel()

    """
    Below are the function calls for the graphs in the document.
    """

    """
    Brownian Motion - vary sigma
    """
    # bot.backtest_bm(s0=100, T=1, bm_sigma=1, dt=1/3600, M=3600, q0=0, gamma=0.01, k=1.5, A=140, plot=True)
    # bot.backtest_bm(s0=100, T=1, bm_sigma=2, dt=1/3600, M=3600, q0=0, gamma=0.01, k=1.5, A=140, plot=True)
    # bot.backtest_bm(s0=100, T=1, bm_sigma=5, dt=1/3600, M=3600, q0=0, gamma=0.001, k=1.5, A=140, plot=True)
    # bot.backtest_bm(s0=100, T=1, bm_sigma=10, dt=1/3600, M=3600, q0=0, gamma=0.001, k=1.5, A=140, plot=True)

    """
    Brownian Motion vary gamma
    """
    # bot.backtest_bm(s0=100, T=1, bm_sigma=2, dt=1/3600, M=3600, q0=0, gamma=0.01, k=1.5, A=140, plot=True)
    # bot.backtest_bm(s0=100, T=1, bm_sigma=2, dt=1/3600, M=3600, q0=0, gamma=0.001, k=1.5, A=140, plot=True)

    """
    Brownian Motion Final PnL Histograms - Vary sigma 
    """
    # bot.plot_bm_pnl_sigma(s0=100, bm_sigmas=[1, 2, 5], T=1, dt=1 / 3600, q0=0, gamma=0.1, k=1.5, A=140, sim_num=100)

    """
    Market Data
    """
    # bot.backtest_data(pair='XBTUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.01, k=5, A=140, plot=True)
    # bot.backtest_data(pair='SOLUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.01, k=5, A=140, plot=True)
    # bot.backtest_data(pair='MATICUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.001, k=5, A=140, plot=True)

    """
    Market Data Appendix
    """
    # bot.backtest_data(pair='ETHUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.001, k=5, A=140, plot=True)
    # bot.backtest_data(pair='ATOMUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.001, k=5, A=140, plot=True)
    # bot.backtest_data(pair='ADAUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.001, k=5, A=140, plot=True)
    # bot.backtest_data(pair='AXSUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.001, k=5, A=140, plot=True)
    # bot.backtest_data(pair='AAVEUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.001, k=5, A=140, plot=True)
    # bot.backtest_data(pair='LINKUSD', T=1, dt=1/3600, M=3600, q0=0, gamma=0.001, k=5, A=140, plot=True)

