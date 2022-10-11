import numpy as np
import numba

@numba.jit(nopython=True, parallel=True)
def compute_fd_parallel(fs, dec=64, k_func=None):
    """
    Computes the Fractal Dimension using the VAR method 
    and `pyramid summing`.

    Parameters
    ----------
    fs : numpy.ndarray
        2D numpy array of the data, shape=(num, N).
    dec : int, optional
        Decimate value for standard k_func. The default is 64.
    k_func : numba.core.registry.CPUDispatcher, optional
        Function that takes N, dec and returns a numpy.ndarray for k_n, 
        must be decorated by @numba.jit(nopython=True).
        By default it is set to use numpy.arange(1, N//(2*dec)) for k_n.

    Returns
    -------
    FDs : numpy.ndarray
        Array of computed Fractal Dimension, shape=(num,).

    """
    FDs = np.empty(shape=fs.shape[0])
    for ii in numba.prange(fs.shape[0]):
        f = fs[ii]
        N = f.shape[0]
        
        k_n = np.arange(1, N//(2*dec)) if k_func is None else k_func(N, dec)
        n_max = len(k_n)

        V_i = np.empty(shape=(n_max))

        u_nk = np.empty(shape=(N-2*k_n[0])) # current iteration
        b_nk = np.empty(shape=(N-2*k_n[0])) # current iteration
        u_nl = np.empty(shape=(N-2*k_n[0])) # previous iteration
        b_nl = np.empty(shape=(N-2*k_n[0])) # previous iteration
        for i in range(0, N-2*k_n[0]):
            u_nk[i] = np.max(f[i:i+2*k_n[0]+1])
            b_nk[i] = np.min(f[i:i+2*k_n[0]+1])
        V_i[0] = np.mean(u_nk-b_nk)

        for n in range(1,n_max):
            u_nl[:] = u_nk
            b_nl[:] = b_nk
            d = k_n[n] - k_n[n-1]
            for i in range(0, N-2*k_n[n]):
                u_nk[i] = max(u_nl[i], u_nl[i+2*d])
                b_nk[i] = min(b_nl[i], b_nl[i+2*d])
            V_i[n] = np.mean(u_nk[:N-2*k_n[n]]-b_nk[:N-2*k_n[n]])

        X = np.log(k_n)
        X_m = np.mean(X)
        Y = np.log(V_i)
        Y_m = np.mean(Y)
        FDs[ii] = 2- np.sum((X-X_m)*(Y-Y_m))/np.sum((X-X_m)**2)
    return FDs

compute_fd_serial = numba.jit(compute_fd_parallel.py_func, parallel=False)

