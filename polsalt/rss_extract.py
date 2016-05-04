import os, sys, glob, shutil, inspect

import numpy as np
import pyfits as pf
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from scipy.ndimage import convolve1d
from scipy import linalg as la
import pylab as P
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval

#import reddir
#datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

from oksmooth import boxsmooth1d,blksmooth2d
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

from scrunch1d import scrunch1d

# np.seterr(invalid='raise')
np.set_printoptions(threshold=np.nan)
debug = True

def spatial_prof_estimate(skysubimg):

    """This function creates an estimate of the average spatial profile given a
       sysubtracted image's 2d data array."""

    #Get rid of nans
    skysubimg[np.isnan(skysubimg)] = 0
    
    #Normalize by spatial direction
    norm_prof = skysubimg[:,:]/np.nansum(skysubimg,axis=0)
    
    #Average by wavelength
    spat_prof = np.nanmean(norm_prof,axis=1)


    fig = P.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(np.arange(spat_prof.shape[0]),spat_prof, 'b-')
    
    ax.set_xlabel('Rows')
    ax.set_ylabel('Profile')
    P.tight_layout()
    P.show()

    return spat_prof

def plot_dumb_extraction(twod_spec, aperture, mask):
    """This function plots the sum/mean/median of a 2D spectrum in the
       given aperture, given some masking."""


    #Mask 2D spectrum
    spec = np.zeros(twod_spec.shape)
    spec[~mask] = twod_spec[~mask]
    spec[np.isnan(spec)] = 0.0

    #Sum/Mean/Median in Apeture
    spec[~aperture] = 0.0
    oned_spec = np.sum(spec, axis=0)

    #Plot
    fig = P.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(np.arange(oned_spec.shape[0]),oned_spec, 'b-')
    ax.set_ylabel('Signal')
    ax.set_xlabel('Column / Wavelength')
   
    P.tight_layout()
    P.show()
    

def weighted_poly_fit(skysub_img_row, variance_row, line, deg=3, sqr_clip=25, plot_fit=False):

    """This function computes a weighted polynomial fit to the
       sky-subtracted and target weighted spectrum."""

    outliers = np.zeros(skysub_img_row.shape[0]).astype('bool')
    new_outliers = np.ones(skysub_img_row.shape[0]).astype('bool')
    count = 0
    loop = 0
    plot_fit = plot_fit

    while True:

               
        #Fit the row with a weighted polynomial
        coef, stats_list = polyfit(np.arange(skysub_img_row[~outliers].shape[0]),skysub_img_row[~outliers],w=1.0/variance_row[~outliers], deg=deg, full=True)
                
        #Evaluate fit
        fit = polyval(np.arange(skysub_img_row.shape[0]),coef)

        #Outliers are beyond threshold set
        #deviation = ((skysub_img_row - fit)**2)*((flambda)**2/variance_row)
        deviation = ((skysub_img_row - fit)**2)/(variance_row)
        new_outliers = deviation > sqr_clip
              
        #Count if we've gotten same outliers twice
        if set(outliers) == set(new_outliers):
            count += 1

        #If we get same outliers twice stop!
        if count == 2:
            break
        
        #Add the new outliers
        outliers |= new_outliers

        #plot_max_trace_comparison(np.arange(skysub_img_row.shape[0]),40*skysub_img_row , fit, mask=outliers)
    
        loop += 1
        #print "Loop number is %d, Number of outliers is %d"%(loop,np.sum(outliers))

   
    if plot_fit and (fit<0.0).any():
        plot_polyfit_comparison(np.arange(skysub_img_row.shape[0]),skysub_img_row , fit, mask=outliers,line=line,stats_list=stats_list)

    #Set any negative values to zero
    fit[fit<0.0] = 0.0
    
    return fit
    


def plot_image(img, bpm, img_shape):
    """This function plots an image in greyscale"""

    #Generate a matrix to hold image values
    image = np.zeros([img_shape[0],img_shape[1]])

    #Fill in image values with flux values at fit (i.e. good) pixels
    image[~bpm] += img[~bpm]
    
    fig = P.figure()
    ax = fig.add_subplot(111)
    vmin=abs(image).min()
    print "vmin is", vmin
    P.gray()
    
    im = ax.imshow(image,vmin=-226,vmax=261,extent=(0,1581,0,1026))
    P.tight_layout()
    P.show()

def plot_spatial_profile(summed_col, rows, wav1, wav2):
    """This function plots a rouch spatial in greyscale
    using the sum over a few wavelengths"""

    
    fig = P.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(rows,summed_col, 'b-')
    ax.set_xlabel('Rows')
    ax.set_ylabel('Max from summing $\lambda$=%.1f -%.1f at Center Row'%(wav1,wav2))
    P.tight_layout()
    P.show()

def plot_maximum_row(summed_col, rows, wav1, wav2,window,maxval):
    """This function plots up the rough spatial profile summed over several wavelenths
       and shows the window the maximum is found within."""

    fig = P.figure()
    ax = fig.add_subplot(111)
    maxtuple = np.where(summed_col==maxval)
    print "maxtuple is",maxtuple
    ax.plot(rows,summed_col, 'b-')
    ax.axvspan(window[0], window[1], facecolor='0.5', alpha=0.5,label="Rows %d - %d"%(window[0],window[1]))
    ax.text(0.1,0.8,"Max: %.2e at Row %d"%(maxval,maxtuple[0]),transform=ax.transAxes)
    
    ax.set_xlabel('Rows')
    ax.set_ylabel('Max from summing $\lambda$=%.1f -%.1f at Center Row'%(wav1,wav2))
    P.legend(loc=0,frameon=False)
    P.tight_layout()
    P.show()

def plot_aperture_window(img, apmask, img_shape):
    """This function plots the window of the image that has been selected for a spatial profile."""

    #Generate a matrix to hold image values
    image = np.zeros([img_shape[0],img_shape[1]])

    #Fix nans
    img[np.isnan(img)] = 0

    #Fill in image values with flux values at fit (i.e. good) pixels
    image[apmask] += img[apmask]
    
    fig = P.figure()
    ax = fig.add_subplot(111)
    vmin=np.nanmin(image)
    vmax = np.nanmax(image)
    print "vmin is", vmin
    print "vmin is", vmax
    P.gray()
    
    im = ax.imshow(image,vmin=-230,vmax=190,extent=(0,img_shape[1],img_shape[0],0))
    P.tight_layout()
    P.show()

def plot_maximum_wave(cols, maxrows):
    """This function plots the rows that have maxima as a function of column (wavelength)."""

    fig = P.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(cols,maxrows, 'b*')
    ax.set_ylabel('Rows')
    ax.set_xlabel('Column / Wavelength')
   
    P.tight_layout()
    P.show()

def plot_polyfit_comparison(row,fract_flux, fit, mask, line, stats_list):
    """This function plots up the input data for an interpolation
       of maximum as a function of wavelength and compares that to
       the input values that were used in the interpolation"""
    
    fig = P.figure()
    ax = fig.add_subplot(111)
    #print "maxval is", maxval
    #sys.exit(1)

    #Collect covariance matrix
    rss = stats_list[0]
        
    
    if mask == None:
        ax.plot(row,fract_flux, 'b+',label='Signal')
    else:
        ax.plot(row[~mask],fract_flux[~mask], 'b+',label='Signal')
        ax.plot(row[mask],fract_flux[mask], 'rx',label='Masked Signal')
        
    ax.plot(row, fit,'k-',label='Estimate')
    ax.text(0.8,0.8,'RSS: %.3e'%(rss),transform=ax.transAxes,ha='center',va='center')
    ax.set_ylabel('Fractional Flux')
    ax.set_xlabel('Wavelength/Column')
    P.title('Line %d'%(line))
    P.legend(loc=0)
    P.tight_layout()
    P.show()
    

def plot_max_trace_comparison(wave, maxval, estimate_maxval, mask=None):
    """This function plots up the input data for an interpolation
       of maximum as a function of wavelength and compares that to
       the input values that were used in the interpolation"""
    
    fig = P.figure()
    ax = fig.add_subplot(111)
    #print "maxval is", maxval
    #sys.exit(1)

    if mask == None:
        ax.plot(wave,maxval, 'b+',label='Signal')
    else:
        ax.plot(wave[~mask],maxval[~mask], 'b+',label='Signal')
        ax.plot(wave[mask],maxval[mask], 'r*',label='Masked Signal')
        
    ax.plot(wave, estimate_maxval,'k-',label='Estimate')
    ax.set_ylabel('Maximum Value')
    ax.set_xlabel('Wavelength')
    P.legend(loc=0)
    P.tight_layout()
    P.show()

def plot_max_smooth_trace_comparison(wave, estimate_maxval,smoothed_maxval):
    """This function plots the raw and smoothed fit to the maximum value trace."""

    fig = P.figure()
    ax = fig.add_subplot(111)
    #print "maxval is", maxval
    #sys.exit(1)
    
    ax.plot(wave,estimate_maxval, 'ks')
    ax.plot(wave,smoothed_maxval ,'r-')
    ax.set_ylabel('Maximum Value')
    ax.set_xlabel('Wavelength')
   
    P.tight_layout()
    P.show()

def plot_max_fits(img, img_shape, mask):
    """This function plots a 2D image of the fits to the maximum
       estimated by interpolating the maxima at all wavelengths"""

    #Generate a matrix to hold image values
    image = np.zeros([img_shape[0],img_shape[1]])

    #Fill in image values with flux values at fit (i.e. good) pixels
    image[mask] += img[mask]

    #Fix nans
    image[np.isnan(image)] = 0
    

    fig = P.figure()
    ax = fig.add_subplot(111)
    vmin = np.nanmin(image)
    med = np.median(image)
    sd = np.std(image,ddof=1)
    vmax = np.nanmax(image)
    
    print "vmin is", vmin
    print "vmax is",vmax
    P.gray()
    
    im = ax.imshow(image,vmin=vmin,vmax=vmax,extent=(0,1581,0,1026))
    P.tight_layout()
    P.show()

def plot_2d_profile(img, img_shape):
    """This function plots a 2D image of the provided profile"""

    #Generate a matrix to hold image values
       
    fig = P.figure()
    ax = fig.add_subplot(111)
    P.gray()
    
    im = ax.imshow(img,vmin=0,vmax=1,extent=(0,img_shape[1],0,img_shape[0]))
    P.tight_layout()
    P.show()

def plot_profile(rows, profile, mask=None):
    """This function plots the 1D spatial profile"""

    fig = P.figure()
    ax = fig.add_subplot(111)
    if mask == None:
        ax.plot(rows,profile, 'g-')
    else:
        ax.plot(rows[mask],profile[mask], 'g*',label='Valid Pixels')
        ax.plot(rows[mask],profile[mask], 'g-',label='Valid Pixels')
        ax.plot(rows[~mask],profile[~mask], 'rx',label='Masked Pixels')
    ax.set_ylabel('Profile')
    ax.set_xlabel('Rows')

    P.legend(loc=0)
    P.tight_layout()
    P.show()
    

def specpolsignalmap(hdu,logfile="log.file",debug=False,missing_skyext=False, npix=20, norm_max=True):
    """
    Find FOV edge, Second order, Littrow ghost
    Find sky lines and produce sky flat for 2d sky subtraction
    Make smoothed psf weighting factor for optimized extraction

    Parameters
    ----------
    hdu: fits.HDUList
       image to be cleaned and used as prep for extraction

    logfile: str
       Output 

    """

    with logging(logfile, debug) as log:

        
        if missing_skyext:
            sci_orc = hdu['skysub.x'].data.copy()
        else:
            sci_orc = hdu['skysub.opt'].data.copy()


        #OJO The variance plane seems to be in e-, so divide by gain squared
        g = 1.6 #e-/DN
        var_orc = hdu['var'].data.copy()/g**2
        badbin_orc = (hdu['bpm'].data > 0)
        wav_orc = hdu['wavelength'].data.copy()

        # trace spectrum, compute spatial profile 
        rows,cols = sci_orc.shape #rows -> spatial, cols -> wavelength
        sciname = hdu[0].header["OBJECT"]
        cbin,rbin = np.array(hdu[0].header["CCDSUM"].split(" ")).astype(int)
        slitid = hdu[0].header["MASKID"]
        if slitid[0] =="P": slitwidth = float(slitid[2:5])/10.
        else: slitwidth = float(slitid)         # in arcsec

        print "Image rows = %d, cols = %d, sciname = %s, cbin, rbin = (%d,%d), slitwidth is : %.2f"%(rows,cols,sciname,cbin,rbin,slitwidth)
        profsmoothfac = 2.5                     # profile smoothed by slitwidth*profsmoothfac
        lam_c = wav_orc[rows/2,:] #Center of spatial direction; wavelength array

        #profile_orc = np.zeros_like(sci_orc)
        profile_orc = np.zeros(sci_orc.shape)
        profilesm_orc = np.zeros_like(sci_orc)
        drow_oc = np.zeros((2,cols))
        maxrow_oc = np.zeros((cols),dtype=int)
        maxval_oc = np.zeros((cols),dtype='float32')
        col_cr,row_cr = np.indices(sci_orc.T.shape) #2D Indices for the image Transpose was neccesary--don't know why

        #Spatial profile summed over large range of wavelengths, size of rows
        wave_window = (cols/2-cols/16,cols/2+cols/16)
        cross_or = np.sum(sci_orc[:,wave_window[0]:wave_window[1]],axis=1) #Summed over wavelengths 1/8*no. of wavelengths?
        #plot_spatial_profile(summed_col=cross_or, rows=np.arange(rows), wav1=wav_orc[rows/2,wave_window[0]], wav2=wav_orc[rows/2,wave_window[1]])
        
        okprof_oyc = np.ones((rows,cols),dtype='bool')
        okprofsm_oyc = np.ones((rows,cols),dtype='bool')
        profile_oyc = np.zeros_like(profile_orc)
        profilesm_oyc = np.zeros_like(profile_orc)
        badbin_oyc = np.zeros_like(badbin_orc)
        sci_oyc = np.zeros_like(sci_orc)
        var_oyc = np.zeros_like(var_orc)
        wav_oyc = np.zeros_like(wav_orc)
        badbinnew_oyc = np.zeros_like(badbin_orc)

        # find spectrum roughly from max of central cut, then within narrow curved aperture
        #Returns spatial value of central wavelengths?
        expectrow = np.zeros((cols),dtype='float32') + rows/2
        okwav_c = np.ones(lam_c.shape,dtype='bool')
        goodcols = okwav_c.sum() #Number of "good" wavelengths

        print "Number of good wavelengths is", goodcols
        
        #Find maximum signal along slit in window of 200 unbinned pixels centered on expected center
        unbinpix_wind = (100,100)
        window = (rows/2-unbinpix_wind[0]/rbin,rows/2+unbinpix_wind[1]/rbin)
                          
        crossmaxval = np.max(cross_or[window[0]:window[1]])
        #plot_maximum_row(summed_col=cross_or, rows=np.arange(rows),wav1=wav_orc[rows/2,wave_window[0]], wav2=wav_orc[rows/2,wave_window[1]],window=window,maxval=crossmaxval)
        
        #Find distance between maximum signal along slit and expected center
        print "Found max in column summed data at row:", np.where(cross_or==crossmaxval)[0][0]
        drow = np.where(cross_or==crossmaxval)[0][0] - rows/2

        #Derive expected row numebrs for entire profile
        row_c = (expectrow + drow).astype(int)
        
        #Find rows that lie within 20 unbinned pixels of valid prediction
        aperture_wind = (12,12)
        isaper_cr = ((row_cr-row_c[:,None])>=-aperture_wind[0]/rbin) & ((row_cr-row_c[:,None])<=aperture_wind[1]/rbin) & okwav_c[:,None]

        #plot_aperture_window(img=sci_orc, apmask=isaper_cr.T,img_shape=sci_orc.shape)

        #Finds row index of maximum
        maxrow_oc[okwav_c] = np.argmax(sci_orc.T[isaper_cr].reshape((goodcols,-1)),axis=1)+ row_c[okwav_c] - aperture_wind[0]/rbin
                
        #Finds max value at that indices pointing to the maxima inside our prev. defined window
        maxval_oc[okwav_c] = sci_orc[maxrow_oc[okwav_c],okwav_c]
        #plot_maximum_wave(wav_orc[maxrow_oc[okwav_c],okwav_c],maxval_oc[okwav_c])
        
        #Distance maximum in spatial direction is from maximum found using wavelength binned profile
        #drow1_c = maxrow_oc - row_c
       
        
        #Spatial location of maximum at central wavelength
        trow_o = maxrow_oc[cols/2]
        print "Trow_o (row where max is at central wavelength) is",trow_o
        drow1_c = maxrow_oc - trow_o

        # divide out spectrum (allowing for spectral curvature) to make spatial profile

        #Find profile defined by maximum flagged OK
        okprof_c = (maxval_oc != 0) & okwav_c

        #Fit Distance that maximum in spatial direction is from maximum found using
        #wavelength binned profile with a polynomial
        order_trace = 1
        drow2_c = np.polyval(np.polyfit(np.where(okprof_c)[0],drow1_c[okprof_c],order_trace),(range(cols)))
        print "drow2_c is", drow2_c
        plot_max_trace_comparison(wave=wav_orc[trow_o,okprof_c], maxval=drow1_c[okprof_c], estimate_maxval=drow2_c[okprof_c])
        #sys.exit(1)

        #BPM EDIT
        #Flags points less than 3 rows different than the poly. fit as good
        err_pix = 3
        okprof_c[okwav_c] &= np.abs(drow2_c - drow1_c)[okwav_c] < err_pix

        #plot_profile(rows=wav_orc[trow_o,:], profile=(okprof_c).astype('int'))
        #sys.exit(1)

        norm_rc = np.zeros((rows,cols))
        norm_yc = np.zeros((rows,cols))
        normsm_rc = np.zeros((rows,cols))

        #Fit the maximum value as a function wavelength using data from maximum at central wavelength
        #maxval_fit_center_lambda = interp1d(wav_orc[trow_o,okprof_c],maxval_oc[okprof_c],bounds_error = False , fill_value=0.)
        #Choose only central row
        maxval_fit_center_lambda = interp1d(wav_orc[trow_o,okprof_c],sci_orc[trow_o,okprof_c],bounds_error = False , fill_value=0.)

        #Fill in maximum value for each wavelength using fit
        for r in range(rows):
            norm_rc[r] = maxval_fit_center_lambda(wav_orc[r])
            
        #Find indices of the 'Rows' rows centered on target
        
        samp_rows = (525 + np.arange(3)-(3-1)/2).astype(int)
        print "samp_rows are:",samp_rows

        #Flag estimated points that are zero as not OK
        #BPM EDIT
        okprof_rc = (norm_rc != 0.)
        #plot_2d_profile(img=okprof_rc.astype('int'), img_shape=okprof_rc.shape)
        #sys.exit(1)
        
        
        #plot_max_fits(norm_rc, norm_rc.shape, mask=okprof_rc)
        #sys.exit(1)
        
        # make a slitwidth smoothed norm and profile for the background area
        for r in range(rows):
            normsm_rc[r] = boxsmooth1d(norm_rc[r],okprof_rc[r],(8.*slitwidth*profsmoothfac)/cbin,0.5)

               
        #Flag smooth points that are 0.0 as not OK
        okprofsm_rc = (normsm_rc != 0.)
        
        #Normalize science frame by the maximum at each wavelength
        #Will result in 1 at the central maximum of 2D spectra, <1 everywhere else
        #Important
        profile_orc[okprof_rc] = sci_orc[okprof_rc]/norm_rc[okprof_rc]
        #profile_orc = sci_orc/norm_rc
        #plot_max_fits(profile_orc, profile_orc.shape, mask=okprof_rc)
        #plot_max_fits(norm_rc, norm_rc.shape, mask=okprof_rc)

        #Save profile
        hduprofn = pf.PrimaryHDU(header=hdu[0].header)   
        hduprofn = pf.HDUList(hduprofn)  
        header=hdu['SCI'].header.copy()       
        hduprofn.append(pf.ImageHDU(data=(norm_rc), header=header, name='NORMMAX'))
        hduprofn.append(pf.ImageHDU(data=(profile_orc), header=header, name='UNSHIFT_PROF_NORMMAX'))
        hduprofn.writeto('normmax_raw.fits',clobber=True,output_verify='warn')

        #Normalize science frame by the smoothed maximum at each wavelength
        #Will result in 1 at the central maximum of 2D spectra, <1 everywhere else
        profilesm_orc[okprofsm_rc] = sci_orc[okprofsm_rc]/normsm_rc[okprofsm_rc]
        #plot_max_fits(profilesm_orc, profilesm_orc.shape, mask=okprofsm_rc)
       
                
        #Distance maxima (as function of columns/wavelength) are from maximum of central column/wavelength 
        #drow_oc = (expectrow - expectrow[cols/2] + drow2_c -drow2_c[cols/2])
        #drow_oc = row_c + drow2_c - row_c[cols/2] - drow2_c[cols/2]
        #drow_oc = maxrow_oc - maxrow_oc[cols/2]
        #drow_oc = (drow2_c - drow2_c[cols/2])
        #drow_oc = drow2_c #also known as the fit to maxrow_oc - trow_o
        drow_oc = maxrow_oc - maxrow_oc#maxrow_oc - trow_o
        #plot_profile(rows=wav_orc[trow_o,:], profile=drow_oc)
        #sys.exit(1)
        
        # take out profile spatial curvature and tilt (r -> y)
       
        #Threshold value for points from spline interpolation of bad pixel mask
        bad_val = 0.1
        
        #Aligns maxima to the same row as the central column/wavelength maximum
        
        for c in range(cols):
            profile_oyc[:,c] = shift(profile_orc[:,c],-drow_oc[c],order=0)
            profilesm_oyc[:,c] = shift(profilesm_orc[:,c],-drow_oc[c],order=0)
            #Flag all new shifted (spline interpolated) bad pixels with values greater than bad_val as bad
            badbin_oyc[:,c] = shift(badbin_orc[:,c].astype(int),-drow_oc[c],cval=1,order=0) > bad_val
            sci_oyc[:,c] = shift(sci_orc[:,c],-drow_oc[c],order=0)
            var_oyc[:,c] = shift(var_orc[:,c],-drow_oc[c],order=0)
            wav_oyc[:,c] = shift(wav_orc[:,c],-drow_oc[c],order=0)
            norm_yc[:,c] = shift(norm_rc[:,c],-drow_oc[c],order=0)

        #Create new masks that eliminate updated bad pixel mask after shifting
        #BPM EDIT
        okprof_oyc = ~badbin_oyc & okprof_rc
        #plot_2d_profile(img=okprof_oyc.astype('int'), img_shape=okprof_oyc.shape)
        #sys.exit(1)
        
        okprofsm_oyc = ~badbin_oyc & okprofsm_rc
        
        
        #Sets wavelength values not on the spatial edges to 0 if the row
        #(spatial direction) before or after was 0 (posisbly from being outside boundary of spline above).
        # square off edge        
        wav_oyc[1:rows-1,:] *= np.logical_not((wav_oyc[1:rows-1,:]>0) & (wav_oyc[:rows-2,:]*wav_oyc[2:rows,:]==0)).astype(int)               

        #Compute median over sample of columns (wavelengths) to form profile as function of row (spatial direction)
        #Note axis =-1 selects the last axis dim in array
        profile_oy = np.median(profilesm_oyc,axis=-1)

        #Plot dumb extraction spectrum
        #plot_dumb_extraction(sci_orc, isaper_cr.T, mask=~okprof_oyc)
        #sys.exit(1)
        
        #Try forming polynomial image
        sci_oyc[np.isnan(sci_oyc)] = 0.0
        var_oyc[np.isnan(var_oyc)] = 0.0
        flambda = np.nansum(sci_oyc,axis=0)
        flambda[np.isnan(flambda)] = 0.0
        poly_prof = np.zeros(sci_oyc.shape)

        targeted = True
        #npix = 60
        target_window = (trow_o + np.arange(npix)-(npix+1)/2).astype(int)
        flambda = np.nansum(sci_orc[target_window,:],axis=0)
        #plot_2d_profile(img=sci_orc/flambda, img_shape=sci_orc.shape)
        

        #Calculate variance in profile image
        var_prof_oyc = np.zeros_like(var_oyc)
        if norm_max:
            #Normalize variance frame by maximum at each wavelength squared
            #var_prof_orc[okprof_rc] = (var_orc[okprof_rc]/g**2)/norm_rc[okprof_rc]**2 #Assumes this is electrons?
            var_prof_oyc = (var_oyc)/norm_yc**2
        else:
            #Normalize variance by a sample extracted spectrum
            var_prof_orc = var_orc/flambda**2

        #Get rid of NaNs
        var_prof_oyc[np.isnan(var_prof_oyc)] = 0.0


        #Save profile
        hduprofa = pf.PrimaryHDU(header=hdu[0].header)   
        hduprofa = pf.HDUList(hduprofa)  
        header=hdu['SCI'].header.copy()       
        hduprofa.append(pf.ImageHDU(data=(profile_oyc.astype('float32')), header=header, name='PROFILE-RAW-NORMMAX'))
        hduprofa.append(pf.ImageHDU(data=(sci_oyc/flambda), header=header, name='PROFILE-RAW-NORMSPEC'))
        hduprofa.writeto('profile_raw.fits',clobber=True,output_verify='warn')
        #sys.exit(1)
        
        #Make polynomial profile
        if not norm_max:
            for r in (range(rows) if not targeted else target_window):
                mask = var_prof_oyc[r,:] == 0
                poly_prof[r,~mask] = weighted_poly_fit(sci_oyc[r,~mask]/flambda[~mask], var_prof_oyc[r,~mask],deg=2, sqr_clip=25**2,plot_fit=True,line=r)
        else:
            for r in (range(rows) if not targeted else target_window):
                mask = var_prof_oyc[r,:] == 0
                poly_prof[r,~mask] = weighted_poly_fit(profile_oyc[r,~mask], var_prof_oyc[r,~mask],deg=2, sqr_clip=25**2,plot_fit=True,line=r)

        #Normalize by wavelength
        profile_oyc = poly_prof/np.sum(poly_prof,axis=0)

        #Print smoothed profiles
        pf.PrimaryHDU(profile_oyc.astype('float32')).writeto(sciname+"_profile_%s_oyc.fits"%('normmax' if norm_max else 'normspec'),clobber=True) 

        #plot_2d_profile(img=(poly_prof!=0).astype('int'), img_shape=poly_prof.shape)
        sys.exit(1)
        

        #/!\ Try just using a 1D profile for all wavelengths
        #spat_prof  = spatial_prof_estimate(skysubimg=sci_orc)
        
        #for c in range(cols):
            #profile_oyc[:,c] = spat_prof
            
        
        # Take FOV from wavmap
        edgerow_od = np.zeros((2))
        badrow_oy = np.zeros((rows),dtype=bool)

        #Finds first place wavelength at center column is positive
        #And (after reversing ordering using ::-1) last place wavelength is positive
        edgerow_od[0] = np.argmax(wav_oyc[:,cols/2] > 0.)
        edgerow_od[1] = rows-np.argmax((wav_oyc[:,cols/2] > 0.)[::-1])

        #Flags everything outside the edges as bad
        badrow_oy = ((np.arange(rows)<edgerow_od[0]) | (np.arange(rows)>edgerow_od[1]))

        #Mid-point between found edges
        axisrow_o = edgerow_od.mean(axis=0)

        #Add newly flagged pixels to masks
        #BPM Edit
        okprof_oyc[badrow_oy,:] = False            
        badbinnew_oyc[badrow_oy,:] = True
        #plot_2d_profile(img=(~badbinnew_oyc).astype('int'), img_shape=badbinnew_oyc.shape)
        #sys.exit(1)

       
       
        log.message('Optical axis row:   %4i ' % (axisrow_o), with_header=False)
        log.message('Target center row: O    %4i' % (trow_o), with_header=False)
        log.message('Bottom, top row:   O %4i %4i \n' \
                % tuple(edgerow_od.flatten()), with_header=False)

        #plot_image(sci_orc, bpm=badbinnew_oyc, img_shape=sci_orc.shape)
       
       
                
        # Mask off atmospheric A- and B-band (profile only) No need
        #ABband = np.array([[7582.,7667.],[6856.,6892.]])         
        #for b in (0,1): okprof_oyc &= ~((wav_oyc>ABband[b,0])&(wav_oyc<ABband[b,1]))

        profile_oyc *= okprof_oyc
        profilesm_oyc *= okprofsm_oyc

        #plot_profile(rows=np.arange(rows), profile=profile_oy)
        
        #plot_2d_profile(profilesm_oyc, profilesm_oyc.shape)
        #sys.exit(1)
        
        # Stray light search by looking for seeing-sized features in spatial and spectral profile                   

        #Select the central window (profile_wind[1] - profile_wind[0]) of odd number so there's a center
        #at the target row
        profile_wind = (16,17)
        profile_Y = profile_oy[trow_o-profile_wind[0]:trow_o+profile_wind[1]] 

        #Fineness of interpolation in profile window
        step_size = 2**-4
        #Interpolate values in chosen central window given desired fineness
        profile_Y = interp1d(np.arange(-1.0*profile_wind[0],1.0*profile_wind[1]),profile_Y,kind='cubic')(np.arange(-1.0*profile_wind[0],1.0*profile_wind[0],step_size))

        #ghost search kernel is size of 3*fwhm and sums to zero
        #Calculate full width half max using argmax to get index/row values of the
        #half max points.
        #N.B. [256:0:-1] indexes elements 256:0 backwards to find most positive <0.5 point first
        half_pt = np.int(profile_wind[0]/step_size)
        fwhm = 3.*(np.argmax(profile_Y[half_pt:]<0.5) + np.argmax(profile_Y[half_pt:0:-1]<0.5))*step_size

        #Define kernel that looks for features 3*fwhm in size
        kernelcenter = np.ones(np.around(fwhm/2)*2+2)
        kernelbkg = np.ones(kernelcenter.shape[0]+4)
        kernel = -kernelbkg*kernelcenter.sum()/(kernelbkg.sum()-kernelcenter.sum())
        kernel[2:-2] = kernelcenter                

        # First, look for second order as feature in spatial direction** No Need***
        #Convolve ghost seeking kernel with each row (spatial direction)
        ghost_oyc = convolve1d(profilesm_oyc,kernel,axis=0,mode='constant',cval=0.)

        #Construct bad pixel map for image used to search for ghost using previously created masks
        isbadghost_oyc = (~okprofsm_oyc | badbin_oyc | badbinnew_oyc)     

        #Convolve bad pixel map of ghost search image with unity kernel 
        #Add any pixels that are not zero after convolution to bad mask
        isbadghost_oyc |= convolve1d(isbadghost_oyc.astype(int),kernelbkg,axis=0,mode='constant',cval=1) != 0

        #Add any pixels within 3 FWHM to bad pixel mask
        isbadghost_oyc[trow_o-3*fwhm/2:trow_o+3*fwhm/2,:] = True

        #Generate image of features that have been masked as ghosts
        #Bad pixels = 1.0, Good pixels= 0.0
        ghost_oyc *= (~isbadghost_oyc).astype(int)                
        stdghost_oc = np.std(ghost_oyc,axis=0)
        boxbins = (int(2.5*fwhm)/2)*2 + 1
        boxrange = np.arange(-int(boxbins/2),int(boxbins/2)+1)
        #plot_2d_profile(ghost_oyc, ghost_oyc.shape)
        #sys.exit(1)
               
        # Remaining ghosts have same position O and E. _Y = row around target, both beams

        #Number of rows between target and edges
        Rows = int(2*np.abs(trow_o-edgerow_od).min()+1)

        #Find indices of the 'Rows' rows centered on target
        row_oY = (trow_o + np.arange(Rows)-(Rows+1)/2).astype(int)
        
        #Select these rows to form the following 2D arrays
        ghost_Yc = ghost_oyc[row_oY,:]
        isbadghost_Yc = isbadghost_oyc[row_oY,:]
        stdghost_c = np.std(ghost_Yc,axis=0)
        profile_Yc = profilesm_oyc[row_oY,:] 
        okprof_Yc = okprof_oyc[row_oY,:]               
                    
        # Search for Littrow ghost as undispersed object off target
        # Convolve with ghost kernal in spectral direction, divide by standard deviation, 
        #  then add up those > 10 sigma within fwhm box

        #Make bad pixel mask for littrow ghost search
        isbadlitt_Yc = isbadghost_Yc | \
            (convolve1d(isbadghost_Yc.astype(int),kernelbkg,axis=1,mode='constant',cval=1) != 0)
        #Convolve kernel with ghost search image and multiply bad pixels by 0
        litt_Yc = convolve1d(ghost_Yc,kernel,axis=-1,mode='constant',cval=0.)*(~isbadlitt_Yc).astype(int)
        #Divide by standard deviation
        litt_Yc[:,stdghost_c>0] /= stdghost_c[stdghost_c>0]
        #Set pixels with signal to noise less than 10 to zero
        litt_Yc[litt_Yc < 10.] = 0.

        #Convolve by column (wavelength) and row (spatial direction) using unity kernel
        for c in range(cols):
            #Fill array with convolution from center of boxbin kernel for Rows rows
            litt_Yc[:,c] = np.convolve(litt_Yc[:,c],np.ones(boxbins))[boxbins/2:boxbins/2+Rows] 
        for Y in range(Rows):
            #Fill array with convolution from center of boxbin kernel for all cols columns
            litt_Yc[Y] = np.convolve(litt_Yc[Y],np.ones(boxbins))[boxbins/2:boxbins/2+cols] 

        #Save index of the maximum
        Rowlitt,collitt = np.argwhere(litt_Yc == np.nanmax(litt_Yc[:cols]))[0]
        #Define 2D indices of the maximum
        littbox_Yc = np.meshgrid(boxrange+Rowlitt,boxrange+collitt)

        # Mask off Littrow ghost (profile and image)
        if litt_Yc[Rowlitt,collitt] > 100:
            islitt_oyc = np.zeros((rows,cols),dtype=bool)

            #Compute median spatial profile by row
            for y in np.arange(edgerow_od[0],edgerow_od[1]):
                #Skip this row if there aren't any unmasked pixels
                if profilesm_oyc[y,okprof_oyc[y,:]].shape[0] == 0: continue
                profile_oy[y] = np.median(profilesm_oyc[y,okprof_oyc[y]])

            #Compute difference between median profile and 2D smooth profile
            dprofile_yc = profilesm_oyc - profile_oy[:,None]
            #Setup box that indexes littrow maximum found earlier
            littbox_yc = np.meshgrid(boxrange+Rowlitt-Rows/2+trow_o,boxrange+collitt)
            
            #Flag points in littrow box that are factor of 10 above noise compared to median profile
            islitt_oyc[littbox_yc] = dprofile_yc[littbox_yc] > 10.*np.sqrt(var_oyc[littbox_yc])

            #If there are mod. S/N points in Littrow box...
            if islitt_oyc.sum():
                #Find their strength and wavelength
                wavlitt = wav_oyc[trow_o,collitt]
                strengthlitt = dprofile_yc[littbox_yc].max()
                #Flag points and add to bad pixell masks
                okprof_oyc[islitt_oyc] = False
                badbinnew_oyc |= islitt_oyc
                isbadghost_Yc[littbox_Yc] = True
                ghost_Yc[littbox_Yc] = 0.
            
                log.message('Littrow ghost masked, strength %7.4f, ypos %5.1f", wavel %7.1f' \
                        % (strengthlitt,(Rowlitt-Rows/2)*(rbin/8.),wavlitt), with_header=False)
    

        # Anything left as spatial profile feature is assumed to be neighbor non-target stellar spectrum
        # Mask off spectra above a threshhold

        #Retrieve mask (ok values) from 'Rows' rows around target
        okprof_Yc = okprof_oyc[row_oY,:]
        #Create mask that selects each row that has any valid/OK points
        okprof_Y = okprof_Yc.any(axis=1)
        #Spatial profile of 'Rows' rows centered on target
        profile_Y = np.zeros(Rows,dtype='float32')

        #Find the median spatial profile for each row that had any valid/OK pixels
        for Y in np.where(okprof_Y)[0]:
            profile_Y[Y] = np.median(profile_Yc[Y,okprof_Yc[Y]])

        #Define row (half FWHM plus 5 pix)
        #Why this row?
        avoid = int(np.around(fwhm/2)) +5
        print "Avoid is", avoid
        
        #Mask off the first and last avoid pixels in addition to the central 2*avoid pixels around target

        okprof_Y[range(avoid) + range(Rows/2-avoid,Rows/2+avoid) + range(Rows-avoid,Rows)] = False
        #print "Sum of okprof is",okprof_Y.sum()
        #plot_profile(np.arange(Rows), profile_Y, mask=okprof_Y)
        #sys.exit(1)

        #Convolve Rows-length median profile with previously defined kernel
        nbr_Y = convolve1d(profile_Y,kernel,mode='constant',cval=0.)
        #plot_profile(np.arange(Rows), nbr_Y, mask=okprof_Y)

        #Define ratio threshold to search for other peaks
        nbrmask = 1.0
        count = 0
        
        #Define maximum number of peaks to mask
        max_peaks = 20
        print "comp term is",np.nanmax(nbr_Y[okprof_Y]/profile_Y[okprof_Y])

        #While the convolved median profile (ignoring target) is greater than the median profile (ignoring target)
        while np.nanmax(nbr_Y[okprof_Y]/profile_Y[okprof_Y]) > nbrmask:
            count += 1
            #Find location of the possible peak that showed up in convolved profile
            nbrrat_Y = np.zeros(Rows)
            nbrrat_Y[okprof_Y] = nbr_Y[okprof_Y]/profile_Y[okprof_Y]
            Ynbr = np.where(nbrrat_Y == np.nanmax(nbrrat_Y))[0]
           
            #Define mask
            #Find first index in reverse order that is below threshold
            Ymask1 = Ynbr - np.argmax(nbrrat_Y[Ynbr::-1] < nbrmask)
            #Find first index past peak that is below threshold
            Ymask2 = Ynbr + np.argmax(nbrrat_Y[Ynbr:] < nbrmask)

            #Record strength relative to target
            strengthnbr = nbr_Y[Ynbr]/nbr_Y[Rows/2]

            #Add peak to bad pixel masks
            okprof_Y[Ymask1:Ymask2+1] = False
            badbinnew_oyc[row_oY[Ymask1:Ymask2],:] = True 
            okprof_oyc[row_oY[Ymask1:Ymask2],:] = False

            log.message('Neighbor spectrum masked: strength %7.4f, ypos %5.1f' \
                            % (strengthnbr,(Ynbr-Rows/2)*(rbin/8.)), with_header=False)
            if count>max_peaks: break

        if debug: np.savetxt(sciname+"_nbrdata_Y.txt",np.vstack((profile_Y,nbr_Y,okprof_Y.astype(int))).T,fmt="%8.5f %8.5f %3i")
        #plot_profile(np.arange(Rows), profile_Y, mask=okprof_Y)
        #sys.exit(1)

        #BPM Edit
        #After neighbor masking
        #plot_2d_profile(img=(~badbinnew_oyc).astype('int'), img_shape=badbinnew_oyc.shape)
        #plot_aperture_window(img=sci_orc, apmask=~badbinnew_oyc, img_shape=sci_orc.shape)
        #sys.exit(1)
        
        #Mark only values with positive wavelength as OK
        okprof_oyc &= (wav_oyc > 0.)

        #Mark only values with valid values as OK
        okprof_oyc &= ~np.isnan(wav_oyc)

        #BPM Edit
        #Masking negative wavelengths
        #plot_2d_profile(img=(okprof_oyc).astype('int'), img_shape=okprof_oyc.shape)
        #plot_aperture_window(img=sci_orc, apmask=okprof_oyc, img_shape=sci_orc.shape)
        #sys.exit(1)
        
        #Save smoothed profile, masked variance map, bad pixel mask, and wavelength map to file
        hduprof = pf.PrimaryHDU(header=hdu[0].header)   
        hduprof = pf.HDUList(hduprof)  
        header=hdu['SCI'].header.copy()       
        hduprof.append(pf.ImageHDU(data=profilesm_oyc.astype('float32'), header=header, name='SCI'))
        hduprof.append(pf.ImageHDU(data=(var_oyc*okprof_oyc).astype('float32'), header=header, name='VAR'))
        hduprof.append(pf.ImageHDU(data=(~okprof_oyc).astype('uint8'), header=header, name='BPM'))
        hduprof.append(pf.ImageHDU(data=wav_oyc.astype('float32'), header=header, name='WAV'))

        if debug: hduprof.writeto(sciname+"_skyflatprof.fits",clobber=True)

        #compute stellar psf in original geometry for extraction
        # use profile normed to (unsmoothed) stellar spectrum, new badpixmap, removing background continuum 

        #Set good pixels to True denoting background continuum? Yep but only relevant to bright stars
        isbkgcont_oyc = ~(badbinnew_oyc)

        #Initialize new arrays
        targetrow_od = np.zeros((2))   
        badbinnew_orc = np.zeros_like(badbin_orc)
        isbkgcont_orc = np.zeros_like(badbin_orc)
        psf_orc = np.zeros_like(profile_orc)

        #Define parameters of 2D smoothing
        rblk = 1
        cblk = int(cols/16) #Perhaps to 1 to only have 1 block.
        target_window = (trow_o + np.arange(3)-(3+1)/2).astype(int)
        #okprof_oyc[:,:] = False
        #okprof_oyc[target_window,:] = True

        #plot_profile(wav_orc[rows/2,:],np.mean(profile_oyc[target_window,:],axis=0), mask=None)
        #sys.exit(1)
        

        #plot_aperture_window(img=profile_oyc, apmask=okprof_oyc, img_shape=profile_oyc.shape)
        #sys.exit(1)
        #Smooth profile using defined block binning
        #Check profile before smoothing for dips/masking
        #/!\ Temporarily disabled the 2d smoothing since bad resid messed this up
        #profile_oyc = blksmooth2d(profile_oyc,okprof_oyc,rblk,cblk,blklim=0.25,mode='median')
        #plot_aperture_window(img=profile_oyc, apmask=okprof_oyc, img_shape=profile_oyc.shape)
        #sys.exit(1)
        
        bad_val = 0.1
        for c in range(cols):
            #Move 2d smoothed profile back to initial position and save as PSF
            psf_orc[:,c] = shift(profile_oyc[:,c],drow_oc[c],cval=0,order=1)
            #Shift masks as well, flagging as bad any pixels with interpolated values > bad_val
            isbkgcont_orc[:,c] = shift(isbkgcont_oyc[:,c].astype(int),drow_oc[c],cval=0,order=1) > bad_val
            badbinnew_orc[:,c] = shift(badbinnew_oyc[:,c].astype(int),drow_oc[c],cval=1,order=1) > bad_val

        #BPM Edit
        #After masking shifted bad pix
        #plot_2d_profile(img=(~badbinnew_orc).astype('int'), img_shape=badbinnew_orc.shape)
        #plot_aperture_window(img=sci_orc, apmask=(~badbinnew_orc), img_shape=sci_orc.shape)
        #sys.exit(1)

        #Find limits of target row using the location of the nearest 'good' points
        #at the central wavelength on either side of the target, starting with the target row
        #N.B. If target row is valid/not marked bad these both will be the target row index
        targetrow_od[0] = trow_o - np.argmax(isbkgcont_orc[trow_o::-1,cols/2] > 0)
        targetrow_od[1] = trow_o + np.argmax(isbkgcont_orc[trow_o:,cols/2] > 0)

        #This array ends up setting row limits for the target
        #This initializes the array with the extent of the rows and location of target in rowspace
        maprow_od = np.vstack((edgerow_od[0],targetrow_od[0],targetrow_od[1],edgerow_od[1])).T

        #This defines the buffer that will be used to extract the target
        nrows = 3
        nbuff = (nrows -1)/2 + 1
        #This buffer is added to limits defined above
        maprow_od += np.array([-nbuff,-nbuff,nbuff,nbuff])
        
        #plot_2d_profile(psf_orc,psf_orc.shape)
        #sys.exit(1)
        
        if debug:
            pf.PrimaryHDU(psf_orc.astype('float32')).writeto(sciname+"_psf_orc.fits",clobber=True) 
            #pf.PrimaryHDU(skyflat_orc.astype('float32')).writeto(sciname+"_skyflat_orc.fits",clobber=True)
            pf.PrimaryHDU(badbinnew_orc.astype('uint8')).writeto(sciname+"_badbinnew_orc.fits",clobber=True) 
            pf.PrimaryHDU(isbkgcont_orc.astype('uint8')).writeto(sciname+"_isbkgcont_orc.fits",clobber=True)           
        return psf_orc,badbinnew_orc,isbkgcont_orc,maprow_od,drow_oc,sciname,wav_orc


def specpolextract(image, hdu, obsname, wav_orc, maprow_od, drow_oc, badbinnew_orc,isbkgcont_orc, psf_orc,debug=True,missing_skyext=False, npix=20):
    """Extract and rebin spectra."""

    #Setup output name
    
    outfile = image.split('.fits')[0]+'_specpolextract.fits'
    
    #Set up dimensions of image
    rows,cols = wav_orc.shape
    
    #Create mask of all valid wavelengths
    okwav_oc = ~((wav_orc == 0).all(axis=0))

    
    #Make a copy of the row parameters for each wavelength
    maprow_ocd = maprow_od + np.zeros((cols,4))
   
    #Add the distance to the maximum to each row; this should now host row parameters
    #with respect to position of the maximum
    #Recall drow_oc is distance maxima (as function of columns/wavelength) are from maximum of central column/wavelength 
    maprow_ocd[okwav_oc] += drow_oc[okwav_oc,None]      

    #Use limits in maprow to define edge of 2D spectrum
    isedge_orc = (np.arange(rows)[:,None] < maprow_ocd[None,:,0]) | \
                 (np.arange(rows)[:,None] > maprow_ocd[None,:,3])

    #Use limits in map row to define target spectrum to extract
    istarget_orc = okwav_oc[None,:] & (np.arange(rows)[:,None] > maprow_ocd[None,:,1]) & \
                   (np.arange(rows)[:,None] < maprow_ocd[None,:,2])
    
    isbkgcont_orc &= (~badbinnew_orc & ~isedge_orc & ~istarget_orc)
    badbinall_orc = badbinnew_orc
    badbinone_orc = badbinnew_orc
    #hdusum['BPM'].data = badbinnew_orc.astype('uint8')

    if debug: 
        #                hdusum.writeto(obsname+".fits",clobber=True)
        pf.PrimaryHDU(psf_orc.astype('float32')).writeto(obsname+'_psf_orc.fits',clobber=True) 
        #               pf.PrimaryHDU(badbinnew_orc.astype('uint8')).writeto('badbinnew_orc.fits',clobber=True)   
        #               pf.PrimaryHDU(badbinall_orc.astype('uint8')).writeto('badbinall_orc.fits',clobber=True)  
        #               pf.PrimaryHDU(badbinone_orc.astype('uint8')).writeto('badbinone_orc.fits',clobber=True)  

    # set up wavelength binning
    wbin = wav_orc[rows/2,cols/2]-wav_orc[rows/2,cols/2-1] 
    wbin = 2.**(np.rint(np.log2(wbin)))         # bin to nearest power of 2 angstroms
    #Set up min and max wavelengths
    wmin = wav_orc.max(axis=0)[okwav_oc].min()
    wmax = wav_orc.max()

    #Find index of last column with wavelength > 0
    colmax = np.where((wav_orc > 0.).any(axis=0))[0][-1]
    #Store row indicies of the last column 
    row_r = np.where(wav_orc[:,colmax] > 0.)[0]
    #Set maximum wavelength to shortest maximum wavelength in 2d spectrum
    wmax = min(wmax,wav_orc[row_r,colmax].min())

    #Define wavelength bin edges
    wedgemin = wbin*int(wmin/wbin+0.5) + wbin/2.
    wedgemax = wbin*int(wmax/wbin-0.5) + wbin/2.
    wedge_w = np.arange(wedgemin,wedgemax+wbin,wbin)
    wavs = wedge_w.shape[0] - 1
    print "number of wavelength bis is",wavs

    #Initialize array to hold 2D bin edges
    binedge_orw = np.zeros((rows,wavs+1))
    #Define array of 2*row_incl row indices centered on target spectrum
    row_incl = rows/4
    specrow_or = (maprow_od[0,1:3].mean(axis=0) + np.arange(-row_incl,row_incl)).astype(int)
    
    # scrunch and normalize psf from summed images (using badbinone) for optimized extraction
    # psf is normalized so its integral over row is 1.
    psfnormmin = 0.70    # wavelengths with less than this flux in good bins are marked bad
    psf_orw = np.zeros((rows,wavs))

    #Scrunch each of 2*row_incl rows centered on target spectrum
    for r in specrow_or:
        binedge_orw[r] = \
                           interp1d(wav_orc[r,okwav_oc],np.arange(cols)[okwav_oc], \
                                    kind='linear',bounds_error=False)(wedge_w)
        psf_orw[r] = scrunch1d(psf_orc[r],binedge_orw[r])

    if debug: 
        pf.PrimaryHDU(binedge_orw.astype('float32')).writeto(obsname+'_binedge_orw.fits',clobber=True)
        pf.PrimaryHDU(psf_orw.astype('float32')).writeto(obsname+'_psf_orw.fits',clobber=True)

    #Normalize PSF by row
    #Hmmm should we normalize by only the good pixels? Want profile accurate, masking might affect that
    psf_orw /= psf_orw.sum(axis=0)#[None,:]
       
    
    #Ralf's new data is missing this extension:
    if missing_skyext:
        target_orc = hdu['skysub.x'].data
    else:
        sky_orc = hdu['SKYSUB.IMG'].data
        target_orc = hdu['skysub.opt'].data
    sciraw_orc = hdu['sci.raw'].data
    sci_orc = hdu['sci'].data

    #Ojo for Ralf Reduction, variance in electrons
    g = 1.6 #e-/DN
    var_orc = hdu['var'].data/g**2
    
    print "Sci raw at 513,790",sciraw_orc[525,790]
    print "Sci  at 513,790",sci_orc[525,790]
    print "var at 513,790 is",var_orc[525,790]
    sys.exit(1)

    #Mask as zero the bad binned pixels
    badbin_orc = (hdu['BPM'].data > 0)
    badbinbkg_orc = (badbin_orc | badbinnew_orc | isedge_orc | istarget_orc)

    #BPM Edit
    #After masking off edge and target
    #plot_2d_profile(img=(~badbin_orc).astype('int'), img_shape=badbin_orc.shape)

    #Get rid of bad pixels
    target_orc *= (~badbin_orc).astype(int)

    #Get rid of NaNs
    target_orc[np.isnan(target_orc)] = 0.0
    var_orc[np.isnan(var_orc)] = 0.0

    if debug:
        pf.PrimaryHDU(target_orc.astype('float32')).writeto('target_orc.fits',clobber=True)

    #Store variance and bad pixel data
    badbin_orc |= badbinnew_orc

    #BPM Edit
    #Add original BPM to one found in signal mapping
    #plot_2d_profile(img=(~badbin_orc).astype('int'), img_shape=badbin_orc.shape)
    #plot_aperture_window(img=target_orc, apmask=(~badbin_orc), img_shape=target_orc.shape)
    #sys.exit(1)

    # extract spectrum optimally (Horne, PASP 1986)
    target_orw = np.zeros((rows,wavs))
    sky_orw = np.zeros((rows,wavs))   
    var_orw = np.zeros_like(target_orw)
    badbin_orw = np.ones((rows,wavs),dtype='bool')   
    wt_orw = np.zeros_like(target_orw)
    update_var_orw = np.zeros((rows,wavs))
    
    #Identically scrunch the target, variance, and bad bin data
    bad_val = 0.001
    for r in specrow_or:
        if not missing_skyext:
            sky_orw[r] = scrunch1d(sky_orc[r],binedge_orw[r])
        target_orw[r] = scrunch1d(target_orc[r],binedge_orw[r])
        var_orw[r] = scrunch1d(var_orc[r],binedge_orw[r])
        #Flag any pixels that have interpolated BPM values > bad_val as bad
        badbin_orw[r] = scrunch1d(badbin_orc[r].astype(float),binedge_orw[r]) > bad_val
    
    #Add bins with variance at zero to bad pixel mask
    #??? Why would variance be zero after scrunching? Take care of unmasked pixels
    badbin_orw |= (var_orw == 0)

    #BPM Edit
    #After masking pixels with 0 variance or bad scrunching
    #plot_2d_profile(img=(~badbin_orw).astype('int'), img_shape=badbin_orw.shape)
    #plot_aperture_window(img=target_orw, apmask=(~badbin_orw), img_shape=target_orw.shape)
    #sys.exit(1)
    
    #Add rows with less than psfnormin flux to bad pixel mask
    print "Before norm cut, BPM pixels is",np.sum(badbin_orw.astype('int'))
    badbin_orw |= ((psf_orw*(~badbin_orw)).sum(axis=0)[None,:] < psfnormmin)
    #NOTE: I get normalization > 1 and at 0 after using this mask for some rows.
    print "After norm cut, BPM pixels is",np.sum(badbin_orw.astype('int'))
    
    #BPM Edit
    #After flagging pixels with less than the psfnorm min bad
    #plot_2d_profile(img=(~badbin_orw).astype('int'), img_shape=badbin_orw.shape)
    #plot_aperture_window(img=target_orw, apmask=(~badbin_orw), img_shape=target_orw.shape)
    #sys.exit(1)

    
    # use master psf shifted in row to allow for guide errors
    #Get rid of nans
    psf_orw[np.isnan(psf_orw)] = 0
    print "PSF nans", np.isnan(psf_orw).sum()
    print "Target nans",np.isnan(target_orw).sum()
    print "Weight nans",np.isnan(wt_orw).sum()
    print "Bad bin orw nans",np.isnan(badbin_orw).sum()
    
    bother_with_pwidth = False
    
    if bother_with_pwidth:
        pwidth = 2*int(1./np.nanmax(psf_orw))
        print "pwidth is:", pwidth
        ok_w = ((psf_orw*badbin_orw).sum(axis=0) < 0.03/float(pwidth/2)).all(axis=0)
        crosscor_s = np.zeros(pwidth)

        for s in range(pwidth):
            crosscor_s[s] = (psf_orw[s:s-pwidth]*target_orw[pwidth/2:-pwidth/2]*ok_w).sum()

        smax = np.argmax(crosscor_s)
        s_S = np.arange(smax-pwidth/4,smax-pwidth/4+pwidth/2+1)
    
        polycof = la.lstsq(np.vstack((s_S**2,s_S,np.ones_like(s_S))).T,crosscor_s[s_S])[0]
        pshift = -(-0.5*polycof[1]/polycof[0] - pwidth/2) if (polycof[1] != 0.0 and polycof[0] != 0.0) else 0.0
    
        s = int(pshift+pwidth)-pwidth
    else:
        #Ignore shifting the PSF for extraction...for now
        pshift = 0
        s = 0
        
    sfrac = pshift-s
    psfsh_orw = np.zeros_like(psf_orw)
    outrow = np.arange(max(0,s+1),rows-(1+int(abs(pshift)))+max(0,s+1))
    #print "outrow is", np.arange(1,rows)

    #samp_rows = (525 + np.arange(20)-(20-1)/2).astype(int)
    ##print "samp_rows are:",samp_rows
    #outrow = samp_rows

    psfsh_orw[outrow] = (1.-sfrac)*psf_orw[outrow-s] + sfrac*psf_orw[outrow-s-1]
    print "PSF SHIFT nans", np.isnan(psfsh_orw).sum()

    #Define weights as inverse variance a la Horne
    wt_orw[~badbin_orw] = psfsh_orw[~badbin_orw]/var_orw[~badbin_orw] #P/V

    #Estimate variance in extracted spectrum
    #var_ow = (psfsh_orw*wt_orw*(~badbin_orw)).sum(axis=0)
    #badbin_ow = (var_ow == 0)
    #var_ow[~badbin_ow] = 1./var_ow[~badbin_ow]

    #Estimate variance in 2 steps
    var_denom_ow = (psfsh_orw*wt_orw*(~badbin_orw)).sum(axis=0) # Sum M*P^2/V
    badbin_ow = (var_denom_ow == 0)
    var_ow = (psfsh_orw*(~badbin_orw)).sum(axis=0)/var_denom_ow # Sum M*P/ Sum M*P^2/V
    
    print "Var nans",np.isnan(var_ow).sum()

    #Define rectangle to extract
    #npix = 60
    samp_rows = (525 + np.arange(npix)-(npix-1)/2).astype(int)
    samp_mask = np.zeros(wt_orw.shape)
    samp_mask[samp_rows,:] = 1.0
    
    #Estimate the optimum spectrum
    #sci_ow = (target_orw*wt_orw).sum(axis=0)*var_ow
    sci_ow = (target_orw*wt_orw*(~badbin_orw)).sum(axis=0)/var_denom_ow

    sig_to_noise = np.median(sci_ow/np.sqrt(var_ow))
    print "Median S/N is ", sig_to_noise
    
    #Test weights
    #masked_psf =  (psfsh_orw*(wt_orw!=0).astype('int'))*(var_ow!=0).astype('int')
    #norm_psf = masked_psf.sum(axis=0)
    #norm_psf = psf_orw.sum(axis=0)
    #print "Norm is", norm_psf
    #sys.exit(1)

    #Estimate sum in rectangle
    apply_bpm = True
    sci_mask_orw = (target_orw*(wt_orw!=0).astype('int'))*(var_ow!=0).astype('int') if apply_bpm else target_orw
    sci_sum_ow = np.sum(sci_mask_orw,axis=0)
    sci_sum_rect_ow = np.sum(sci_mask_orw*samp_mask,axis=0)

    sig_to_noise = np.median(sci_sum_rect_ow/np.sqrt(np.sum(var_orw*samp_mask,axis=0)))
    print "Median S/N in rect is ", sig_to_noise
    
    #Estimate average
    sci_avg_ow = np.mean(sci_mask_orw,axis=0)
    sci_avg_rect_ow = np.mean(sci_mask_orw*samp_mask,axis=0)

    #Estimate median
    sci_med_ow = np.median(sci_mask_orw,axis=0)
    sci_med_rect_ow = np.median(sci_mask_orw*samp_mask,axis=0)

    #Save image for optimal extraction
    hduout0 = pf.PrimaryHDU(header=hdu[0].header)    
    hduout0 = pf.HDUList(hduout0)
    header0=hdu['SCI'].header.copy()
    hduout0.append(pf.ImageHDU(data=target_orw, header=header0, name='DATA'))
    hduout0.append(pf.ImageHDU(data=(target_orw*wt_orw*(~badbin_orw))/var_denom_ow, header=header0, name='WEIGHTED'))
    hduout0.append(pf.ImageHDU(data=sci_mask_orw, header=header0, name='MASKDATA'))
    hduout0.append(pf.ImageHDU(data=sci_mask_orw*samp_mask, header=header0, name='DATA.RECT'))
    hduout0.writeto('target_'+outfile,clobber=True,output_verify='warn')


    #Test extractions
    #Save image for optimal extraction
    hduout1 = pf.PrimaryHDU(header=hdu[0].header)    
    hduout1 = pf.HDUList(hduout1)
    header1=hdu['SCI'].header.copy()
    header1.update('CRVAL1',wedge_w[0]+wbin/2.)
    header1.update('CRVAL2',0)
    header1.update('CDELT1',wbin)
    header1.update('CTYPE1','Angstroms')
        
    hduout1.append(pf.ImageHDU(data=sci_ow, header=header1, name='OPT'))
    hduout1.append(pf.ImageHDU(data=sci_sum_ow, header=header1, name='SUM'))
    hduout1.append(pf.ImageHDU(data=sci_avg_ow, header=header1, name='AVG'))
    hduout1.append(pf.ImageHDU(data=sci_med_ow, header=header1, name='MED'))   
    hduout1.writeto('test_extract_'+outfile,clobber=True,output_verify='warn')

    
    hduout2 = pf.PrimaryHDU(header=hdu[0].header)    
    hduout2 = pf.HDUList(hduout2)
    header2=hdu['SCI'].header.copy()
    header2.update('CRVAL1',wedge_w[0]+wbin/2.)
    header2.update('CRVAL2',0)
    header2.update('CDELT1',wbin)
    header2.update('CTYPE1','Angstroms')
    header2.update('NPIX',npix)

    hduout2.append(pf.ImageHDU(data=sci_sum_rect_ow, header=header2, name='SUMRECT'))
    hduout2.append(pf.ImageHDU(data=sci_avg_rect_ow, header=header2, name='AVGRECT'))
    hduout2.append(pf.ImageHDU(data=sci_med_rect_ow, header=header2, name='MEDRECT')) 
    hduout2.writeto('test_rect_extract_'+'%02d_pix_'%(npix)+outfile,clobber=True,output_verify='warn')
    sys.exit(1)
            
    #Iterate on optimal extraction
    sigma = 50
    rn = 2.1 #DN from 1st chip noise
    outliers = badbin_orw == True
    loop = 0
    count = 0
    pix_masked = 0
    nonneg = (~(target_orw < 0)).astype('int')
    save_fits = False
    
    while True:
        
        #Update the variance
        update_var_orw = rn**2 + np.abs(sci_ow*psfsh_orw) + np.abs(sky_orw)

        #Mask pixels above a threshold
        deviation = (target_orw - sci_ow*psfsh_orw)**2

        if save_fits:
            #Save image for optimal extraction
            hduout1 = pf.PrimaryHDU(header=hdu[0].header)    
            hduout1 = pf.HDUList(hduout1)
            header1=hdu['SCI'].header.copy()
            header1.update('CRVAL1',wedge_w[0]+wbin/2.)
            header1.update('CRVAL2',0)
            header1.update('CDELT1',wbin)
            header1.update('CTYPE1','Angstroms')
            
            hduout1.append(pf.ImageHDU(data=sci_ow*psfsh_orw, header=header1, name='FIT'))
            hduout1.append(pf.ImageHDU(data=target_orw, header=header1, name='DATA'))
            hduout1.append(pf.ImageHDU(data=deviation, header=header1, name='DEV'))
            #hduout.append(pf.ImageHDU(data=badbin_ow.astype("uint8"), header=header, name='BPM'))            
            hduout1.writeto('fit_'+outfile,clobber=True,output_verify='warn')
            plot_aperture_window(img=deviation, apmask=outrow, img_shape=deviation.shape)
            sys.exit(1)
        
        new_outliers = deviation > sigma*update_var_orw
        
        #Re-calculate spectrum
        mask = (~new_outliers).astype('int')
        print "%d pixels not masked."%(np.sum(mask))
        wt_orw = mask*psfsh_orw/update_var_orw
        norm_ow = 1.0/(wt_orw*psfsh_orw).sum(axis=0)
        sci_ow = (target_orw*wt_orw).sum(axis=0)*norm_ow
        var_sci_ow = (mask*psfsh_orw).sum(axis=0)*norm_ow
                    
        #Count if we've gotten same outliers twice
        if pix_masked  == np.sum(new_outliers.astype('int')):
            count += 1

        #If we get same outliers twice stop!
        if count == 2:
            break

        print "Loop number is", loop
        pix_masked = np.sum(new_outliers.astype('int'))
        loop += 1
        sig_to_noise = np.median(sci_ow/np.sqrt(var_sci_ow))
        print "Median S/N is ", sig_to_noise

    #Add the new outliers
    #outliers |= new_outliers     
        
    #Plot spectrum
    plot_profile(np.linspace(wmin,wmax,wavs), sci_ow)
    
    #Plot image to be extracted
    plot_aperture_window(img=target_orw*wt_orw*var_ow, apmask=outrow, img_shape=target_orw.shape)
    opt = target_orw*wt_orw*norm_ow
        
    #Save image for optimal extraction
    hduout1 = pf.PrimaryHDU(header=hdu[0].header)    
    hduout1 = pf.HDUList(hduout1)
    header1=hdu['SCI'].header.copy()
    header1.update('CRVAL1',wedge_w[0]+wbin/2.)
    header1.update('CRVAL2',0)
    header1.update('CDELT1',wbin)
    header1.update('CTYPE1','Angstroms')
    
    hduout1.append(pf.ImageHDU(data=opt, header=header1, name='OPT'))
    #hduout.append(pf.ImageHDU(data=badbin_ow.astype("uint8"), header=header, name='BPM'))            
    hduout1.writeto('opt_'+outfile,clobber=True,output_verify='warn')

    
    badlim = 0.20
    psfbadfrac_ow = (psfsh_orw*badbin_orw.astype(int)).sum(axis=0)/psfsh_orw.sum(axis=0)
    badbin_ow |= (psfbadfrac_ow > badlim)
    
    # write O,E spectrum, prefix "s". VAR, BPM for each spectrum. y dim is virtual (length 1)
    # for consistency with other modes
    hduout = pf.PrimaryHDU(header=hdu[0].header)    
    hduout = pf.HDUList(hduout)
    header=hdu['SCI'].header.copy()
    header.update('VAREXT',2)
    header.update('BPMEXT',3)
    header.update('CRVAL1',wedge_w[0]+wbin/2.)
    header.update('CRVAL2',0)
    header.update('CDELT1',wbin)
    header.update('CTYPE1','Angstroms')
    
    hduout.append(pf.ImageHDU(data=sci_ow.reshape((1,wavs)), header=header, name='SCI'))
    header.update('SCIEXT',1,'Extension for Science Frame',before='VAREXT')
    hduout.append(pf.ImageHDU(data=var_ow.reshape((1,wavs)), header=header, name='VAR'))
    hduout.append(pf.ImageHDU(data=badbin_ow.astype("uint8").reshape((1,wavs)), header=header, name='BPM'))            
    
    hduout.writeto('e'+outfile,clobber=True,output_verify='warn')
    #log.message('Output file '+'e'+outfile, with_header=False)

    return


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print "Incorrect number of objects. Please supply image and 1 if it's missing sky extensions."
        sys.exit(1)


    image = sys.argv[1]
    missing_skyext = True if sys.argv[2] == '1' else False
    npix = np.int(sys.argv[3])
    hdu = pf.open(image)
    psf_orc,badbinnew_orc,isbkgcont_orc,maprow_od,drow_oc,sciname,wav_orc=specpolsignalmap(hdu,logfile="log.file",debug=False, missing_skyext=missing_skyext,npix=npix)
    specpolextract(image,hdu,sciname, wav_orc, maprow_od, drow_oc, badbinnew_orc,isbkgcont_orc=isbkgcont_orc,missing_skyext=missing_skyext,psf_orc=psf_orc,npix=npix)
