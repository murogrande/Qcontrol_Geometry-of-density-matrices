def delete_less_than_k(arr,k=1.0e-6):
    """Eliminate small numbers in a vector """
    for i in range(len(arr)):#Iterating the list arr
        Z = (arr[i].real, arr[i].imag)
        if abs(Z[0])<k:
            a=0
        else:
            a=Z[0]
        if abs(Z[1])<k:
            b=0.0j
        else:
            b=Z[1]
        
        arr[i]=a+b