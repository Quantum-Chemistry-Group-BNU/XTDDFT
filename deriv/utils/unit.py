# refer to pyscf
# ha2eV = 27.21138602  # pyscf transformation check from 'pyscf/data/nist.py'
# bohr = 0.52917721092
c = 137.03599967994  # atom unit light speed

# refer to ORCA
ha2eV = 27.2113834  # orca transformation
bohr = 0.5291772083
cgs2au = 1/(235.7220 * 2)

# find online
eVxnm = 1239.842  # 1eV * 1nm = 1240
erg2ha = 2.2937 * 1e10  # gauss unit to atom unit, energy
esu2au = 2.0819 * 1e9  # gauss unit to atom unit, electron charge
cm2bohr = 1.8897 * 1e8  # gauss unit to atom unit, length
gauss2au = 4.2544 * 1e-10  # gauss unit to atom unit, magnetic field
# gauss unit to atom unit, circular dichroism spectrum rotatory strength, same with gaussian multiply 1e-40
# cgs2au = erg2ha * esu2au * cm2bohr / gauss2au * 1e-40

# refer to BDF
BDF_c = 137.0359895000
au2D = 2.541765

eV2cm_1 = 8065.545  # eV to cm^{-1}