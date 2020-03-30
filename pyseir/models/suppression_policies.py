import numpy as np
from scipy.interpolate import interp1d


def generate_triggered_suppression_model(t_list, lockdown_days, open_days, reduction=0.25):
    """
    Generates a contact reduction model which switches a binary supression
    policy on and off.

    Parameters
    ----------
    t_list:
    lockdown_days: int
        Days of reduced contact rate.
    open_days:
        Days of high contact rate.

    Returns
    -------
    suppression_model: callable
        suppression_model(t) returns the current suppression model at time t.
    """

    state = 'lockdown'
    state_switch = lockdown_days
    rho = []

    if lockdown_days == 0:
        rho = np.ones(len(t_list))
    elif open_days == 0:
        rho = np.ones(len(t_list)) * reduction
    else:
        for t in t_list:
            if t >= state_switch:
                if state == 'open':
                    state = 'lockdown'
                    state_switch += lockdown_days
                elif state == 'lockdown':
                    state = 'open'
                    state_switch += open_days
            if state == 'open':
                rho.append(1)
            elif state == 'lockdown':
                rho.append(reduction)

    return interp1d(t_list, rho, fill_value='extrapolate')


def piecewise_parametric_policy(x, t_list):
    """
    Generate a piecewise suppression policy over n_days based on interval
    splits at levels passed and according to the split_power_law.

    Parameters
    ----------
    x: array(float)
        x[0]: split_power_law
            The splits are generated based on relative proportions of
            t ** split_power_law. Hence split_power_law = 0 is evenly spaced.
        x[1:]: suppression_levels: array-like
            Series of suppression levels that will be equally strewn across.
    t_list: array-like
        List of days over which the period.

    Returns
    -------
    policy: callable
        Interpolator for the suppression policy.
    """
    split_power_law = x[0]
    suppression_levels = x[1:]
    period = int(np.max(t_list) - np.min(t_list))
    periods = np.array([(t + 1) ** split_power_law for t in range(len(suppression_levels))])
    periods = (periods / periods.sum() * period).cumsum()
    periods[-1] += 0.001  # Prevents floating point errors.
    suppression_levels = [suppression_levels[np.argwhere(t <= periods)[0][0]] for t in t_list]
    policy = interp1d(t_list, suppression_levels, fill_value='extrapolate')
    return policy


def fourier_parametric_policy(x, t_list, suppression_bounds=(0.5, 1.5)):
    """
    Generate a piecewise suppression policy over n_days based on interval
    splits at levels passed and according to the split_power_law.

    Parameters
    ----------
    x: array(float)
        First N coefficients for a Fourier series which becomes inversely
        transformed to generate a real series. a_0 is taken relative to level
        the mean at 0.75 (a0 = 3 * period / 4) * X[0]
    t_list: array-like
        List of days over which the period.
    suppression_bounds: tuple(float)
        Lower and upper bounds on the suppression level. This clips the fourier
        policy.

    Returns
    -------
    policy: callable
        Interpolator for the suppression policy.
    """
    frequency_domain = np.zeros(len(t_list))
    frequency_domain[0] = (3 * (t_list.max() - t_list.min()) / 4) * x[0]
    frequency_domain[1:len(x)] = x[1:]
    time_domain = np.fft.ifft(frequency_domain).real + np.fft.ifft(frequency_domain).imag

    return interp1d(t_list, time_domain.clip(min=suppression_bounds[0], max=suppression_bounds[1]),
                    fill_value='extrapolate')
