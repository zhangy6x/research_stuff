import os, numpy as np, numpy.fft as nfft

# copied from wavepy by APS
def dummy_images(imagetype=None, shape=(100, 100), **kwargs):
    """

    Dummy images for simple tests.


    Parameters
    ----------

    imagetype: str
        See options Below
    shape: tuple
        Shape of the image. Similar to :py:mod:`numpy.shape`
    kwargs:
        keyword arguments depending on the image type.


    Image types

        * Noise (default):    alias for ``np.random.random(shape)``

        * Stripes:            ``kwargs: nLinesH, nLinesV``

        * SumOfHarmonics: image is defined by:
          .. math:: \sum_{ij} Amp_{ij} \cos (2 \pi i y) \cos (2 \pi j x).

            * Note that ``x`` and ``y`` are assumed to in the range [-1, 1].
              The keyword ``kwargs: harmAmpl`` is a 2D list that can
              be used to set the values for Amp_ij, see **Examples**.

        * Shapes: see **Examples**. ``kwargs=noise``, amplitude of noise to be
          added to the image

        * NormalDist: Normal distribution where it is assumed that ``x`` and
          ``y`` are in the interval `[-1,1]`. ``keywords: FWHM_x, FWHM_y``


    Returns
    -------
        2D ndarray


    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(dummy_images())

    is the same than

    >>> plt.imshow(dummy_images('Noise'))


    .. image:: img/dummy_image_Noise.png
       :width: 350px


    >>> plt.imshow(dummy_images('Stripes', nLinesV=5))

    .. image:: img/dummy_image_stripe_V5.png
       :width: 350px


    >>> plt.imshow(dummy_images('Stripes', nLinesH=8))

    .. image:: img/dummy_image_stripe_H8.png
       :width: 350px


    >>> plt.imshow(dummy_images('Checked', nLinesH=8, nLinesV=5))

    .. image:: img/dummy_image_checked_v5_h8.png
       :width: 350px


    >>> plt.imshow(dummy_images('SumOfHarmonics', harmAmpl=[[1,0,1],[0,1,0]]))

    .. image:: img/dummy_image_harmonics_101_010.png
       :width: 350px

    >>> plt.imshow(dummy_images('Shapes', noise = 1))

    .. image:: img/dummy_image_shapes_noise_1.png
       :width: 350px

    >>> plt.imshow(dummy_images('NormalDist', FWHM_x = .5, FWHM_y=1.0))

    .. image:: img/dummy_image_NormalDist.png
       :width: 350px


    """

    if imagetype is None:
        imagetype = 'Noise'

    if imagetype == 'Noise':
        return np.random.random(shape)

    elif imagetype == 'Stripes':
        if 'nLinesH' in kwargs:
            nLinesH = int(kwargs['nLinesH'])
            return np.kron([[1, 0] * nLinesH],
                           np.ones((shape[0], shape[1]/2/nLinesH)))
        elif 'nLinesV':
            nLinesV = int(kwargs['nLinesV'])
            return np.kron([[1], [0]] * nLinesV,
                           np.ones((shape[0]/2/nLinesV, shape[1])))
        else:
            return np.kron([[1], [0]] * 10, np.ones((shape[0]/2/10, shape[1])))

    elif imagetype == 'Checked':

        if 'nLinesH' in kwargs:
            nLinesH = int(kwargs['nLinesH'])

        else:
            nLinesH = 1

        if 'nLinesV' in kwargs:
            nLinesV = int(kwargs['nLinesV'])
        else:
            nLinesV = 1

        _arr = np.ones((int(shape[0]/2/nLinesV), int(shape[1]/2/nLinesH)))
        return np.kron([[1, 0] * nLinesH, [0, 1] * nLinesH] * nLinesV, _arr)
        # Note that the new dimension is int(shape/p)*p !!!

    elif imagetype == 'SumOfHarmonics':

        if 'harmAmpl' in kwargs:
            harmAmpl = kwargs['harmAmpl']
        else:
            harmAmpl = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]

        sumArray = np.zeros(shape)
        iGrid, jGrid = np.mgrid[-1:1:1j * shape[0], -1:1:1j * shape[1]]

        for i in range(len(harmAmpl)):
            for j in range(len(harmAmpl[0])):
                sumArray += harmAmpl[i][j] * np.cos(2 * np.pi * iGrid * i) \
                            * np.cos(2 * np.pi * jGrid * j)

        return sumArray

    elif imagetype == 'Shapes':

        if 'noise' in kwargs:
            noiseAmp = kwargs['noise']
        else:
            noiseAmp = 0.0

        dx, dy = int(shape[0]/10), int(shape[1]/10)
        square = np.ones((dx * 2, dy * 2))
        triangle = np.tril(square)

        array = np.random.rand(shape[0], shape[1]) * noiseAmp

        array[1 * dx:3 * dx, 2 * dy:4 * dy] += triangle
        array[5 * dx:7 * dx, 1 * dy:3 * dy] += triangle * -1

        array[2 * dx:4 * dx, 7 * dy:9 * dy] += np.tril(square, +1)

        array[6 * dx:8 * dx, 5 * dy:7 * dy] += square
        array[7 * dx:9 * dx, 6 * dy:8 * dy] += square * -1

        return array

    elif imagetype == 'NormalDist':

        FWHM_x, FWHM_y = 1.0, 1.0

        if 'FWHM_x' in kwargs:
            FWHM_x = kwargs['FWHM_x']
        if 'FWHM_y' in kwargs:
            FWHM_y = kwargs['FWHM_y']

        x, y = np.mgrid[-1:1:1j * shape[0], -1:1:1j * shape[1]]

        return np.exp(-((x/FWHM_x*2.3548200)**2 +
                        (y/FWHM_y*2.3548200)**2)/2)  # sigma for FWHM = 1

    else:
        print_color("ERROR: image type invalid: " + str(imagetype))

        return np.random.random(shape)

def propTF_RayleighSommerfeld(u1, Lx, Ly, wavelength, z):

    print('Propagation Using RayleighSommerfeld TF')

    (My, Mx)=np.shape(u1)    #get input field array size
    dx=Lx/Mx    #sample interval
    dy=Ly/My    #sample interval
    k=2*np.pi/wavelength #wavenumber

    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Mx)
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,My)     #freq coords
    [FX,FY]=np.meshgrid(fx,fy)

#    H = np.exp(-1j*np.pi*wavelength*z*(FX**2+FY**2))       #trans func
    H = np.exp(1j*k*z*np.sqrt(1.0-(wavelength*FX)**2-(wavelength*FY)**2))
                                        #trans func RayleighSommerfeld
    H = nfft.fftshift(H)     #shift trans func

    U1=nfft.fft2(nfft.fftshift(u1))     #shift, fft src field
    U2=H*U1      #multiply

    u2=nfft.ifftshift(nfft.ifft2(U2)) #inv fft, center obs field

    return u2

def propTF_RayleighSommerfeld_1D(u1, Lx, wavelength, z):
    # print('Propagation Using RayleighSommerfeld TF')
    Mx,=np.shape(u1)    #get input field array size
    dx=Lx/Mx    #sample interval
    k=2*np.pi/wavelength #wavenumber
    FX=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,Mx)
    H = np.exp(1j*k*z*np.sqrt(1.0-(wavelength*FX)**2))
                                        #trans func RayleighSommerfeld
    H = nfft.fftshift(H)     #shift trans func
    U1=nfft.fft(nfft.fftshift(u1))     #shift, fft src field
    U2=H*U1      #multiply
    u2=nfft.ifftshift(nfft.ifft(U2)) #inv fft, center obs field
    return u2

def fresnelNumber(a, d, wavelength, verbose=False):
    nf = a**2/wavelength/d
    if verbose:
        print(('Nf = %.5g' % np.abs(nf)))
        print('Conditions:')
        print('Nf << 1 Fraunhofer regime;')
        print('Nf >> 1 Geometric Optic;')
        print('Nf -> 1 Fresnel Difraction.')
    return nf

