import torch



'''
Original from:
https://github.com/stefanoschmidt1995/GWAnomalyDetection/blob/main/Algorithm/useful_funcs.py
'''
def torch_wasserstein_distance(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(_torch_cdf_distance(tensor_a,tensor_b))


def _torch_cdf_distance(tensor_a,tensor_b):
    """
    Torch implementation of _cdf_distance for Wasserstein distance
    input: tensor_a, tensor_b
    output: cdf_loss which the computed distance between the tensors.
    
    #Note: this function yields an difference of \approx 10^-9
    """
    """
    Updated for batch support | 29/03/2022
    Updated for multivariate time series support | 29/03/2022
        
        Expects tensor_a and tensor_b to be of shape: (batch_size, segment_length, n_features),
        thus a single batch of 10 time series with lengths of 12 would have the shape: (1, 12, 10)
    """
    batch_size = tensor_a.shape[0]
    assert tensor_a.shape == tensor_b.shape, 'tensor_a and tensor_b have different shape'
    
    #It is necessary to reshape the tensors to match the dimensions of Scipy.
    tensor_a=torch.reshape(torch.swapaxes(tensor_a, -1, -2), (batch_size, tensor_a.shape[2], tensor_a.shape[1])) 
    tensor_b=torch.reshape(torch.swapaxes(tensor_b, -1, -2), (batch_size, tensor_b.shape[2], tensor_b.shape[1])) 
    
    # Creater sorters:
    sorter_a=torch.argsort(tensor_a, dim=-1);   sorter_b=torch.argsort(tensor_a, dim=-1)
    
    # We append both tensors and sort them
    all_values=torch.cat((tensor_a, tensor_b), dim=-1)
    all_values, idx = torch.sort(all_values, dim=-1); 
    
    # Calculate the n-th discrete difference along the given axis (equivalent to np.diff())
    deltas=  all_values[:,:, 1:] - all_values[:,:, :-1];
    
    sorted_a, idx = torch.sort(tensor_a, dim=-1); 
    sorted_b, idx = torch.sort(tensor_b, dim=-1); 
    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    a_cdf_index= torch.searchsorted(sorted_a.flatten(start_dim=2), all_values[:,:,:-1],right=True); # TODO: torch.searchsorted() expects contiguousarrays, passing non-contiguousarrays slows performance due to data copy | fix doesn't seem trivial
    b_cdf_index= torch.searchsorted(sorted_b.flatten(start_dim=2), all_values[:,:,:-1],right=True);
    #Compute the cdf
    a_cdf = a_cdf_index/tensor_a.shape[-1]; 
    b_cdf = b_cdf_index/tensor_b.shape[-1];
    #And the distance between them
    cdf_distance = torch.sum(torch.mul(torch.abs((a_cdf-b_cdf)), deltas),dim=-1)
    
    cdf_loss = cdf_distance.mean()
    return cdf_loss
