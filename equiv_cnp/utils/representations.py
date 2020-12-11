def reps_from_ids(gspace, ids):
    """
    Input:
        gspace - the gspace to get the irreps from
        ids - list - elements 0,-11 stand for trivial and regular rep,
                            elements k=1,2,3,4,... for irrep(k) if self.flip is false,
                            elements [l,k] for l=0,1; k=1,2,3,4,... if self.flip is true
    Output: list of irreducible representations
    """
    irreps = []
    for id in ids:
        if id == 0:
            irreps.append(gspace.trivial_repr)
        elif id == -1:
            irreps.append(gspace.regular_repr)
        elif isinstance(id, int):
            irreps.append(gspace.irrep(id))
        else:
            irreps.append(gspace.irrep(*id))

    return irreps