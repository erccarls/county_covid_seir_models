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
