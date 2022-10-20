import numpy as np
import numba

@numba.njit("f8[:](f8[:,:], i4)", parallel=True)
def compute_fd_parallel(fs, dec, step):
    """
    Computes the Fractal Dimension using the VAR method 
    and `pyramid summing` for double-precision (float64) data.
    For single-precision (float32) data use compute_fd_parallel_f4/compute_fd_serial_f4.

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
    FDs = np.empty(shape=fs.shape[0], dtype=np.float64)
    N = fs.shape[1]
    k_n = np.arange(1, N//(2*dec), 1, np.int64)
    n_max = len(k_n)
    for ii in numba.prange(fs.shape[0]):
        f = fs[ii]
        V_i = np.empty(shape=(n_max), dtype=np.float64)
        
        ub = np.empty(shape=(N-2*k_n[0],2), dtype=np.float64) # current iteration
        for i in range(0, N-2*k_n[0]):
            ub[i,0] = np.max(f[i:i+2*k_n[0]+1])
            ub[i,1] = np.min(f[i:i+2*k_n[0]+1])
        V_i[0] = np.mean(ub[:,0]-ub[:,1])

        for n in range(1,n_max):
            d = k_n[n] - k_n[n-1]
            for i in range(0, N-2*k_n[n]):
                ub[i,0] = max(ub[i,0], ub[i+2*d,0])
                ub[i,1] = min(ub[i,1], ub[i+2*d,1])
            V_i[n] = np.mean(ub[:N-2*k_n[n],0]-ub[:N-2*k_n[n],1])

        X = np.log(k_n)
        X_m = X - np.mean(X)
        Y = np.log(V_i)
        Y_m = Y - np.mean(Y)
        FDs[ii] = 2 - np.sum((X_m)*(Y_m))/np.sum((X_m)**2)
    return FDs

@numba.njit("f4[:](f4[:,:], i4)", parallel=True)
def compute_fd_parallel_f4(fs, dec, step):
    """
    Computes the Fractal Dimension using the VAR method 
    and `pyramid summing` for single-precision (float32) data.
    For double-precision (float64) data use compute_fd_parallel/compute_fd_serial.

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
    FDs = np.empty(shape=fs.shape[0], dtype=np.float32)
    N = fs.shape[1]
    k_n = np.arange(1, N//(2*dec), 1, np.int32)
    n_max = len(k_n)
    for ii in numba.prange(fs.shape[0]):
        f = fs[ii]
        V_i = np.empty(shape=(n_max), dtype=np.float32)
        
        ub = np.empty(shape=(N-2*k_n[0],2), dtype=np.float32) # current iteration
        for i in range(0, N-2*k_n[0]):
            ub[i,0] = np.max(f[i:i+2*k_n[0]+1])
            ub[i,1] = np.min(f[i:i+2*k_n[0]+1])
        V_i[0] = np.mean(ub[:,0]-ub[:,1])

        for n in range(1,n_max):
            d = k_n[n] - k_n[n-1]
            for i in range(0, N-2*k_n[n]):
                ub[i,0] = max(ub[i,0], ub[i+2*d,0])
                ub[i,1] = min(ub[i,1], ub[i+2*d,1])
            V_i[n] = np.mean(ub[:N-2*k_n[n],0]-ub[:N-2*k_n[n],1])

        X = np.log(k_n)
        X_m = X - np.mean(X)
        Y = np.log(V_i)
        Y_m = Y - np.mean(Y)
        FDs[ii] = 2 - np.sum((X_m)*(Y_m))/np.sum((X_m)**2)
    return FDs

compute_fd_serial = numba.njit(compute_fd_parallel.py_func, parallel=False)
compute_fd_serial_f4 = numba.njit(compute_fd_parallel_f4.py_func, parallel=False)
