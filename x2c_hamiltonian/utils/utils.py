import pyscf, numpy

def read_ns(mol,S):
    norb = mol.nao
    Smax = int(S*2)
    nc,no,nv = (mol.nelectron-Smax)//2, Smax, mol.nao-(mol.nelectron-Smax)//2-Smax
    slc = slice(0,nc,1)
    slo = slice(nc,nc+no,1)
    slv = slice(nc+no,norb,1)
    assert nc+no+nv==norb
    return norb, nc,no,nv, slc,slo,slv

def read_fso_file(filename):
    '''
    Read fso(in MOs) from BDF package.
    One can 
    '''
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        import re
        match = re.match(r'iop=\s*(\d+)\s+ncomp=\s*(\d+)\s+nbas=\s*(\d+)', first_line)
            
        iop = int(match.group(1))
        ncomp = int(match.group(2))
        nbas = int(match.group(3))
        
        data = []
        for line in f:
            if line.strip():
                data.append(float(line.strip()))
        data = numpy.array(data)
        assert len(data) == ncomp * nbas * nbas
        
        if iop == 0:
            # fso(ncomp, nbas, nbas)
            matrix = data.reshape(ncomp, nbas, nbas, order='F')
            print(f"shape of fso (ncomp, nbas, nbas) = {(ncomp, nbas, nbas)}")
        else:
            # fso(nbas, nbas, ncomp)
            matrix = data.reshape(nbas, nbas, ncomp, order='F')
            print(f"shape of fso (nbas, nbas, ncomp) = {(nbas, nbas, ncomp)}")
        
        return {
            'iop': iop,
            'ncomp': ncomp,
            'nbas': nbas,
            'matrix': matrix,
            'shape': matrix.shape
        }