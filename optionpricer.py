from abc import ABC, abstractmethod
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import scipy.stats as si
import yfinance as yf
from datetime import datetime
from matplotlib import pyplot as plt
import multiprocessing as mp

# S is the spot price of underlined asset
# K is the strike
# T is the maturity
# r is risk-free rate
# sigma Volatility

# Interface
class IPrice(ABC):
    @abstractmethod
    def price(self, K, S, T, r, sigma, option_type):
        pass

# Interface
class IInstrument(ABC):
    @abstractmethod
    async def get_data(self, ticker, start_date):
        pass

    @abstractmethod
    def calculate_histo_volatility(self):
        pass

    @abstractmethod
    def get_spot_price(self):
        pass

# Option abstract class
class AOption(ABC):
    def __init__(self, strike_price, option_type, risk_free_rate):
        self._strike_price = strike_price
        self._time_to_maturity = None
        self._underlying_asset_price = None
        self._volatility = None
        self._option_type = option_type
        self._risk_free_rate = risk_free_rate
        self._premium = None

    @abstractmethod
    async def calculate_premium(self):
        pass
    @property
    def time_to_maturity(self):
        return self._time_to_maturity

    @time_to_maturity.setter
    def time_to_maturity(self, value):
        self._time_to_maturity = value
        
    @property
    def underlying_asset_price(self):
        return self._underlying_asset_price

    @underlying_asset_price.setter
    def underlying_asset_price(self, value):
        self._underlying_asset_price = value

    @property
    def volatility(self):
        return self._volatility

    @volatility.setter
    def volatility(self, value):
        self._volatility = value

    @property
    def premium(self):
        return self._premium

    @premium.setter
    def premium(self, value):
        self._premium = value
        
    @property
    def option_type(self):
        return self._option_type

# concrete class which implement Option abstract class
class VanillaOption(AOption):
    #The dependency injection is done through this constructor.
    def __init__(self, instrument, pricing_model, strike_price, option_type, risk_free_rate, maturity, variety, current_date=datetime.today().strftime("%Y-%m-%d")):
        super().__init__(strike_price, option_type, risk_free_rate)
        self.instrument = instrument
        self.pricing_model = pricing_model
        self.time_to_maturity = self.calculate_time_to_maturity(maturity, current_date)
        self.variety = variety

    # This asynchronous method loads data necessary for the volatility calculation 
    async def load_data(self, ticker, start_date):
        await self.instrument.get_data(ticker, start_date)
        self.underlying_asset_price = self.instrument.get_spot_price()
        self.volatility = self.instrument.calculate_histo_volatility()

    # This method calculates and return the option maturity from its expiration date
    def calculate_time_to_maturity(self, maturity, current_date): # parameter without a default must be before ones with.
        return (datetime.strptime(maturity, "%Y-%m-%d") - datetime.strptime(current_date, "%Y-%m-%d")).days / 365

    # This method calculates the option premium  
    def calculate_premium(self):
        self.premium = self.pricing_model.price(
            self._strike_price, self._underlying_asset_price,
            self._time_to_maturity, self._risk_free_rate,
            self._volatility, self._option_type
        )
        
# concrete class which implement IInstrument interface
class Stock(IInstrument):
    def __init__(self):
        self.data = None

    async def get_data(self, ticker, start_date):
        def download_data(ticker, start_date):
            return yf.download(ticker, start=start_date, end=datetime.today().strftime("%Y-%m-%d"))
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            self.data = await loop.run_in_executor(pool, download_data, ticker, start_date)
        return self.data

    # This method calculates volatility
    def calculate_histo_volatility(self):
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        return np.std(log_returns) * np.sqrt(252)

    # This method retrieve the spot underlined asset price
    def get_spot_price(self):
        self.spot_price = self.data['Close'].iloc[-1]
        return self.spot_price

    # This method plot the graph helping to Visualize the volatility regime of the underlying asset's prices in order to choose an appropriate historical data period for calculating historical volatility.
    def plot_histo_underlined_asset_price(self,ticker, start_date):
        data = yf.download(ticker, start=start_date, end=datetime.today().strftime("%Y-%m-%d"))
        # Create the plot
        plt.figure(figsize=(10, 6)) 
        plt.plot(data.index, data['Close'], color='orange')
        # Tilt the x-axis dates
        plt.xticks(rotation=45)  # Rotate the x-axis labels by 45 degrees

        # Customizing the background and all other elements
        plt.gca().set_facecolor('black')  # Set the background of the plot area to black
        plt.gcf().set_facecolor('black')  # Set the background of the figure to black
        # Adding and customizing axes lines (spines)
        ax = plt.gca()
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        # Customizing the ticks and labels
        plt.tick_params(colors='white')  # Change the tick color to white
        plt.title('Underlined asset historical price visualisation', color='white')
        plt.xlabel('Time', color='white')
        plt.ylabel('Underlined asset price', color='white')
        
        # Customizing the grid (if needed)
        plt.grid(color='gray', linestyle='--', alpha=0.5)  # Optional: Add a gray grid

        # Showing
        plt.show()

class BlackScholesPrice(IPrice):
    def price(self, K, S0, T, r, sigma, option_type):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = (S0 * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        else:
            price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S0 * si.norm.cdf(-d1, 0.0, 1.0))
        return price
        
# the function to be parallelized must not be a local function of the one where you use multiprocessing.
# for example here below simulate_path() must not be in local of price()
class MonteCarloPrice(IPrice):
    def simulate_path(self, n_steps, S0, num_simulations, dt, r, sigma): # 
            S = np.zeros((num_simulations, n_steps + 1))
            S[:, 0] = S0
            #np.random.seed(0) #ensures that the results of the random generation functions are the same each time they are run(useful for debugging)
            for t in range(1, n_steps + 1):
                Z = np.random.normal(0, 1, num_simulations // 2)
                W = np.concatenate((Z, -Z))  # Variance Reduction Techniques: Variables antithétiques
                S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W)
            return S
        
    def price(self, K, S0, T, r, sigma, option_type, num_simulations=100000, n_steps=100):
        
        def plot_paths(X, n_steps):
            t = range(0,n_steps+1,1)
            # Create the plot
            # Customizing the figure size.
            plt.figure(figsize=(10, 6)) #,facecolor='black'
            # Customizing the background and all other elements
            plt.gca().set_facecolor('black')  # Set the background of the plot area to black
            plt.gcf().set_facecolor('black')  # Set the background of the figure to black
            # Adding and customizing axes lines (spines)
            ax = plt.gca()
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            for j in range(0,n_steps-1):
                plt.plot(t,X[j])
            plt.title('Simulation of Underlined asset price paths', color='white')
            plt.xlabel('Time', color='white')
            plt.ylabel('Underlined asset price', color='white')
            # Customizing the grid (if needed)
            plt.grid(color='gray', linestyle='--', alpha=0.5)  # Optional: Add a gray grid
    
            # Showing
            plt.show()
            
        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)
        simulations_per_process = num_simulations // num_processes
        dt = T / n_steps
        
        with mp.Pool(processes=num_processes) as pool:
            results = [pool.apply_async(self.simulate_path, (n_steps, S0, simulations_per_process, dt, r, sigma)) for _ in range(num_processes)]
            paths = [res.get() for res in results]
        
        S = np.concatenate(paths, axis=0)
        plot_paths(S, n_steps)
        S_avg = np.mean(S[:, 1:], axis=1)
        if option_type == 'call':
            payoff = np.maximum(S_avg - K, 0)
        else:
            payoff = np.maximum(K - S_avg, 0)
        price = np.exp(-r * T) * np.mean(payoff)
        return price