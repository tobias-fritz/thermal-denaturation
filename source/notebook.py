#!/usr/bin/env python
import numpy as np
import pandas as pd
import random
from scipy.optimize import least_squares

def sample_data(seed=9470):
    """
    Generate sample thermal denaturation data with added noise.
    
    Arguments:
        seed (int): Seed for random number generation to ensure reproducibility.
    
    Returns:
    numpy.ndarray: Array of generated data points with noise.
    """
    # Set a seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    t = np.linspace(293, 363, num=40)

    # generate random parameters for sigmoidal
    s_a = random.uniform(3, 6)
    m_a = random.uniform(-0.005, 0)
    s_b = random.uniform(3, 6)
    m_b = random.uniform(-0.005, 0)
    d_h = random.uniform(200000, 400000)
    t_m = random.uniform(317, 327)

    # generate the datapoints
    y = (s_a + m_a * t + (s_b + m_b * t) * np.exp(-d_h * (1/t_m - 1/t) / 8.314)) / (1 + np.exp(-d_h * (1/t_m - 1/t) / 8.314))
    
    # add some random noise
    noise = np.random.normal(0, 0.15, len(y))

    return y + noise

def model(t, param):
    """
    Sigmoidal model for thermal denaturation.
    
    Arguments:
        t (numpy.ndarray): Array of temperature values.
        param (list): List of parameters [sa, ma, sb, mb, dH, tm].
    
    Returns:
    numpy.ndarray: Array of model predictions.
    """
    sa, ma, sb, mb, d_h, t_m = param
    return (sa + ma * t + (sb + mb * t) * np.exp(-d_h * (1/t_m - 1/t) / 8.314)) / (1 + np.exp(-d_h * (1/t_m - 1/t) / 8.314))

def loss_function(param, x, y):
    """
    Loss function for least squares optimization.
    
    Arguments:
        param (list): List of parameters [sa, ma, sb, mb, dH, tm].
        x (numpy.ndarray): Array of temperature values.
        y (numpy.ndarray): Array of observed data points.
    
    Returns:
    numpy.ndarray: Array of residuals (observed - predicted).
    """
    # y - y_predicted
    return y - model(x, param)

def fit_data(x, y, initial_params):
    """
    Fit the model to the data using least squares optimization.
    
    Arguments:
        x (numpy.ndarray): Array of temperature values.
        y (numpy.ndarray): Array of observed data points.
        initial_params (list): Initial guess for the parameters [sa, ma, sb, mb, dH, tm].
    
    Returns:
    numpy.ndarray: Optimized parameters.
    """
    # least squares optimizer
    lsq = least_squares(loss_function, initial_params, args=(x, y), verbose=2)
    return lsq.x

def normalize(x):
    """
    Normalize the data to the range [0, 1].
    
    Arguments:
        x (numpy.ndarray): Array of data points.
    
    Returns:
    numpy.ndarray: Normalized data points.
    """
    return (x - x.min()) / (x.max() - x.min())


def denaturation_analysis(fname: str):

    try: 
        data = pd.read_csv(fname)
    except:
        data = None
        print('No data file found')

    if data:
        # Normalize the data
        data['fraction'] = data[['signal']].apply(normalize)

        # initial parameters
        param = [ 3, -0.003, 4,  -0.003, 4e+05, 325]

        # least squares optimizer
        lsq = least_squares(loss_function, param, args=(data['temperature / K'], data['fraction']), verbose=2)

        # save the fit in the dataframe
        data['fit'] = data['temperature / K'].apply(model,param = lsq.x)

        # Plot the data and the fit
        ax = data.plot.scatter('temperature / °C','fraction', color = 'red', label = 'DATA')
        data.plot('temperature / °C', 'fit', color = 'k',label='FIT', ax=ax);

        # Print the enthalpy and midpoint temperature
        print(f'Enthalpy: {lsq.x[4]} J/mol')
        print(f'Midpoint temperature: {lsq.x[5]} K')