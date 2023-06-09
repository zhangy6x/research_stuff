import os, numpy as np
import waveprop

v0 = 2200 # m/s
lambda0 = 3.956/(v0/1e3)  # wavelength corresponding to 2200m/s

def absorption_Gd():
    'absorption coefficient of Gd: mu. T=exp(-mu*L) '
    sigma0_Gd = 49700 # barn
    density_Gd = 7.90 # g/cm3
    amu = 1.66e-27 # kg
    return sigma0_Gd*1e-28*density_Gd*1e-3/1e-6/(157.25*amu)
mu_Gd = absorption_Gd()

nSLD_Si = np.pi/37e-6/4.1e-10

# !! probably should add duty cycle of G1 as a parameter
def nGI_planewave(
        wavelength = 3.35e-10,
        d01= 6.4, d12 = 2.379e-2,
        g1_thickness = 45e-6,
        g1_period = 7.97e-6, np_g1 = 64, # number of points for the grating 1 period
        npoints = 2**12, # total number of points to simulate transversally
        nSLD = nSLD_Si,
        verbose = False,
):
    """This function calculates propagation of a source through a phase grating and then to a detector position. For the source this function does not consider the repetitive nature G0 grating, but rather just one slit. And the wave incident on the G1 is considered as plane wave.
    """
    Lx = npoints*g1_period/np_g1
    zt = (g1_period/2)**2/wavelength
    X = np.linspace(-Lx/2, Lx/2, npoints)
    if verbose:
        print('Lx = {:.3f} um'.format(Lx*1e6))
        print('npoints period = {:d}'.format(np_g1))
        print('npoints total = {:d}'.format(npoints))
        print('grating lateral size = {:.2f} um'.format(g1_period*np_g1*1e6))
    # source
    sigma_x = 1e-3
    # emSource = ((1j)*X*0.0 +1.0)*np.exp(-X**2/sigma_x**2)
    emSource = (1j)*X*0.0 +1.0
    sourceStr = 'planeWave'
    # grid
    Nrows = npoints//np_g1
    gr1_binary = np.zeros((Nrows, np_g1))
    gr1_binary[:, :np_g1//2] = 1
    gr1_binary.shape = -1,
    marginx = (npoints - gr1_binary.shape[0])//2
    gr1_binary = np.pad(gr1_binary, ((marginx, marginx),), mode='constant', constant_values=(1,))
    gr1_binary = np.pad(gr1_binary, ((0, npoints - gr1_binary.shape[0]),), mode='constant', constant_values=(1,))
    # phase grid
    gr1 = np.exp(-1j*gr1_binary*wavelength*g1_thickness*nSLD) # Pi phase grating
    #gr1 = np.exp(-1j*np.pi/2*gr1_binary) # Pi/2 phase grating
    # propagate to before phase grating
    if verbose:
        waveprop.fresnelNumber(g1_period, 6., wavelength, verbose=True)
    b4_gr1 = waveprop.propTF_RayleighSommerfeld_1D(emSource, Lx, wavelength, d01)
    # after phase grating
    after_gr1 = b4_gr1 * gr1
    # propagate to detector
    if verbose:
        waveprop.fresnelNumber(g1_period, d12, wavelength, verbose=True)
    b4_gr2 = waveprop.propTF_RayleighSommerfeld_1D(after_gr1,Lx, wavelength, d12)
    return b4_gr1, gr1_binary, gr1, after_gr1, b4_gr2

def nGI_spherical(
        wavelength = 3.35e-10,
        d01= 6.4, d12 = 2.379e-2,
        g1_thickness = 45e-6,
        g1_period = 7.97e-6, np_g1 = 64, # number of points for the grating 1 period
        npoints = 2**12, # total number of points to simulate transversally
        nSLD = nSLD_Si,
        verbose = False,
):
    """This function calculates propagation of a source through a phase grating and then to a detector position. For the source this function does not consider the repetitive nature G0 grating, but rather just one slit. And the wave incident on the G1 is considered as a spherical wave.
    """
    Lx = npoints*g1_period/np_g1
    zt = (g1_period/2)**2/wavelength
    X = np.linspace(-Lx/2, Lx/2, npoints)
    if verbose:
        print('Lx = {:.3f} um'.format(Lx*1e6))
        print('npoints period = {:d}'.format(np_g1))
        print('npoints total = {:d}'.format(npoints))
        print('grating lateral size = {:.2f} um'.format(g1_period*np_g1*1e6))
    # wave in front of g1
    r = np.sqrt(d01**2+X**2)
    k = 2*np.pi/wavelength
    phase = k*r
    b4_gr1 = np.cos(phase)+1j*np.sin(phase) #/r
    Nrows = npoints//np_g1
    gr1_binary = np.zeros((Nrows, np_g1))
    gr1_binary[:, :np_g1//2] = 1
    gr1_binary.shape = -1,
    marginx = (npoints - gr1_binary.shape[0])//2
    gr1_binary = np.pad(gr1_binary, ((marginx, marginx),), mode='constant', constant_values=(1,))
    gr1_binary = np.pad(gr1_binary, ((0, npoints - gr1_binary.shape[0]),), mode='constant', constant_values=(1,))
    # phase grid
    gr1 = np.exp(-1j*gr1_binary*wavelength*g1_thickness*nSLD) # Pi phase grating
    #gr1 = np.exp(-1j*np.pi/2*gr1_binary) # Pi/2 phase grating
    # after phase grating
    after_gr1 = b4_gr1 * gr1
    # propagate to detector
    if verbose:
        waveprop.fresnelNumber(g1_period, d12, wavelength, verbose=True)
    b4_gr2 = waveprop.propTF_RayleighSommerfeld_1D(after_gr1,Lx, wavelength, d12)
    return b4_gr1, gr1_binary, gr1, after_gr1, b4_gr2

def calc_visibitilty(
        wl,
        g0_slit_width, g0_period, g0_thickness = 10e-6, mu_g0 = mu_Gd,
        g1_period = 7.97e-6, g1_thickness = 45e-6, np_g1 = 64,
        g2_slit_width = 2e-6, g2_period = 4e-6, g2_thickness = 6e-6,
        d01= 6.4, d12 = 2.379e-2,
        verbose=False,
        **kwds):
    """
    return:
        { x, y0 (without convolution),
          y_G0_proj (G0 projection on G2), y_G2 (G2),
          y0_X_G0 (convolved with G0 projection), y0_X_G0_X_G2 (convolved with G0 then G2),
          V (visibitilty) }
    """
    b4_gr1, gr1_binary, gr1, after_gr1, b4_gr2 = nGI_spherical(
        wavelength = wl, g1_period=g1_period, g1_thickness=g1_thickness,
        np_g1=np_g1, d01=d01, d12=d12,
        **kwds)
    # sl = slice(npoints//2-np_g1*10, npoints//2+np_g1*10)
    sl = slice(None, None)
    y0 = b4_gr2[sl]
    N = y0.size
    dx = g1_period/np_g1
    x = np.arange(-(N-1.)/2*dx, (N-1.)/2*dx+0.1*dx, dx)
    # G0 projectec
    Tr = np.exp(-mu_g0*wl/lambda0*1e10*g0_thickness) # transmission
    if verbose:
        print("transmission:", Tr)
    projected_g0_slit_width = g0_slit_width*d12/d01
    projected_g0_period = g0_period*d12/d01
    N1 = int((x[-1]-x[0])//projected_g0_period)
    y_G0_proj = np.ones(x.size)*Tr
    for i in range(N1+1):
        xstart = x[0] + projected_g0_period*i
        xstart = min(xstart, x[-1])
        xend = xstart + projected_g0_slit_width
        mask = (x>=xstart)*(x<xend)
        y_G0_proj[mask] = 1.
    y_G0_proj/=np.sum(y_G0_proj)
    y0_X_G0 = np.convolve(np.abs(y0)**2, y_G0_proj, mode='same')
    # y0_X_G0 = np.convolve(y0, y_G0_proj, mode='same')
    # y0_X_G0 = np.abs(y0_X_G0)**2
    # G2
    Tr = np.exp(-mu_g0*wl/lambda0*1e10*g2_thickness) # transmission
    y_G2 = np.ones(y0.shape)*Tr
    np_g2 = int(round(g2_period/g1_period*np_g1))
    N_g2_periods = int(np.ceil(y0.shape[0]//np_g2))
    np_slit = int(round(g2_slit_width/g1_period*np_g1))
    for ig2p in range(N_g2_periods):
        y_G2[ig2p*np_g2: ig2p*np_g2+np_slit] = 1
    y_G2/=np.sum(y_G2)
    y0_X_G0_X_G2 = y3 = np.convolve(y0_X_G0, y_G2, mode='same')
    y3_middle = y3[N//2-np_g1: N//2+np_g1]
    ymax, ymin = np.max(y3_middle), np.min(y3_middle)
    V = (ymax-ymin)/(ymax+ymin)
    return x, y0, y_G0_proj, y_G2, y0_X_G0, y0_X_G0_X_G2, V
