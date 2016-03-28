import os, sys, glob, shutil, inspect

import numpy as np
import pyfits as pf
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from scipy.ndimage import convolve1d
from scipy import linalg as la
import pylab as P

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

    #Fill in image values with flux values at fit (i.e. good) pixels
    image[apmask] += img[apmask]
    
    fig = P.figure()
    ax = fig.add_subplot(111)
    vmin=abs(image).min()
    print "vmin is", vmin
    P.gray()
    
    im = ax.imshow(image,vmin=-226,vmax=261,extent=(0,1581,0,1026))
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

def plot_max_trace_comparison(wave, maxval, estimate_maxval):
    """This function plots up the input data for an interpolation
       of maximum as a function of wavelength and compares that to
       the input values that were used in the interpolation"""
    
    fig = P.figure()
    ax = fig.add_subplot(111)
    #print "maxval is", maxval
    #sys.exit(1)
    
    ax.plot(wave,maxval, 'b+')
    ax.plot(wave, estimate_maxval,'k-')
    ax.set_ylabel('Maximum Value')
    ax.set_xlabel('Wavelength')
   
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

    fig = P.figure()
    ax = fig.add_subplot(111)
    vmin = np.nanmin(image)
    med = np.median(image)
    sd = np.std(image,ddof=1)
    vmax = np.nanmax(image)
    
    print "vmin is", vmin
    print "vmax is",vmax
    P.gray()
    
    im = ax.imshow(image,vmin=1e-1,vmax=1,extent=(0,1581,0,1026))
    P.tight_layout()
    P.show()

def plot_2d_profile(img, img_shape):
    """This function plots a 2D image of the provided profile"""

    #Generate a matrix to hold image values
       
    fig = P.figure()
    ax = fig.add_subplot(111)
    
   
    P.gray()
    
    im = ax.imshow(img,vmin=0.1,vmax=1,extent=(0,img_shape[1],0,img_shape[0]))
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
        ax.plot(rows[~mask],profile[~mask], 'rx',label='Masked Pixels')
    ax.set_ylabel('Profile')
    ax.set_xlabel('Rows')

    P.legend(loc=0)
    P.tight_layout()
    P.show()
    

def specpolsignalmap(hdu,logfile="log.file",debug=False):
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
        
        sci_orc = hdu['skysub.opt'].data.copy()
        var_orc = hdu['var'].data.copy()
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

        profile_orc = np.zeros_like(sci_orc)
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
        drow = np.where(cross_or==crossmaxval)[0][0] - rows/2

        #Derive expected row numebrs for entire profile
        row_c = (expectrow + drow).astype(int)
               
        #Find rows that lie within 20 unbinned pixels of valid prediction
        aperture_wind = (20,20)
        #print "Row cr is",row_cr.shape
        #print "row_c is", row_c[:,None].shape
        #print "row_cr - row_c is",(row_cr-row_c[:,None]).shape
        #sys.exit(1)
                                   
        
        isaper_cr = ((row_cr-row_c[:,None])>=-aperture_wind[0]/rbin) & ((row_cr-row_c[:,None])<=aperture_wind[1]/rbin) & okwav_c[:,None]

        #plot_aperture_window(img=sci_orc, apmask=isaper_cr.T,img_shape=sci_orc.shape)

        #Finds row index of maximum
        maxrow_oc[okwav_c] = np.argmax(sci_orc.T[isaper_cr].reshape((goodcols,-1)),axis=1)+ row_c[okwav_c] - aperture_wind[0]/rbin
        #print "maxrow is",maxrow_oc[okwav_c]
        
        #Finds value at that index
        maxval_oc[okwav_c] = sci_orc[maxrow_oc[okwav_c],okwav_c]
        #sys.exit(1)
        #plot_maximum_wave(wav_orc[maxrow_oc[okwav_c],okwav_c],maxrow_oc[okwav_c])
        
        #Distance maximum in spatial direction is from maximum found using wavelength binned profile
        drow1_c = maxrow_oc - (rows/2 + drow)

        #Spatial location of maximum at central wavelength
        trow_o = maxrow_oc[cols/2]

        # divide out spectrum (allowing for spectral curvature) to make spatial profile

        #Find profile defined by maximum flagged OK
        okprof_c = (maxval_oc != 0) & okwav_c

        #Fit Distance that maximum in spatial direction is from maximum found using
        #wavelength binned profile with a 3rd order polynomial
        order_trace = 3
        drow2_c = np.polyval(np.polyfit(np.where(okprof_c)[0],drow1_c[okprof_c],order_trace),(range(cols)))

        #Flags points less than 3 rows different than the poly. fit as good
        err_pix = 3
        okprof_c[okwav_c] &= np.abs(drow2_c - drow1_c)[okwav_c] < err_pix

        norm_rc = np.zeros((rows,cols))
        normsm_rc = np.zeros((rows,cols))

        #Fit the maximum value as a function wavelength using data from maximum at central wavelength
        maxval_fit_center_lambda = interp1d(wav_orc[trow_o,okprof_c],maxval_oc[okprof_c],bounds_error = False , fill_value=0.)

        #Fill in maximum value for each wavelength using fit
        for r in range(rows):
            norm_rc[r] = maxval_fit_center_lambda(wav_orc[r])
            
        
        #plot_max_trace_comparison(wave=np.arange(cols)[okprof_c], maxval=maxval_oc[okprof_c], estimate_maxval=norm_rc[trow_o+10,okprof_c])
        #sys.exit(1)

        #Flag estimated points that are zero as not OK
        okprof_rc = (norm_rc != 0.)
        #plot_max_fits(norm_rc, norm_rc.shape, mask=okprof_rc)
        #sys.exit(1)
        
        # make a slitwidth smoothed norm and profile for the background area
        for r in range(rows):
            normsm_rc[r] = boxsmooth1d(norm_rc[r],okprof_rc[r],(8.*slitwidth*profsmoothfac)/cbin,0.5)

        #Flag smooth points that are 0.0 as not OK
        okprofsm_rc = (normsm_rc != 0.)
        #print "wave indexed size is",wav_orc[okprofsm_rc].shape

        #plot_max_fits(normsm_rc, normsm_rc.shape, mask=okprofsm_rc)
        #sys.exit(1)
        #sys.exit(1)
        #plot_max_smooth_trace_comparison(wave=wav_orc[okprofsm_rc], estimate_maxval=norm_rc[okprofsm_rc],smoothed_maxval=normsm_rc[okprofsm_rc])
    
        #Normalize science frame by the maximum at each wavelength
        #Will result in 1 at the central maximum of 2D spectra, <1 everywhere else
        profile_orc[okprof_rc] = sci_orc[okprof_rc]/norm_rc[okprof_rc]
        #plot_max_fits(profile_orc, profile_orc.shape, mask=okprof_rc)
        #sys.exit(1)

        #Normalize science frame by the smoothed maximum at each wavelength
        #Will result in 1 at the central maximum of 2D spectra, <1 everywhere else
        
        profilesm_orc[okprofsm_rc] = sci_orc[okprofsm_rc]/normsm_rc[okprofsm_rc]
        #plot_max_fits(profilesm_orc, profilesm_orc.shape, mask=okprofsm_rc)
        #sys.exit(1)

        #Normalize variance frame by maximum at each wavelength squared
        var_orc[okprof_rc] = var_orc[okprof_rc]/norm_rc[okprof_rc]**2 #Assumes this is electrons?

        #Distance maxima (as function of columns/wavelength) are from maximum of central column/wavelength 
        drow_oc = (expectrow - expectrow[cols/2] + drow2_c -drow2_c[cols/2])

        # take out profile spatial curvature and tilt (r -> y)
       

        #Threshold value for points from spline interpolation of bad pixel mask
        bad_val = 0.1
        
        #Aligns maxima to the same row as the central column/wavelength maximum
        for c in range(cols):
            profile_oyc[:,c] = shift(profile_orc[:,c],-drow_oc[c],order=1)
            profilesm_oyc[:,c] = shift(profilesm_orc[:,c],-drow_oc[c],order=1)
            #Flag all new shifted (spline interpolated) bad pixels with values greater than bad_val as bad
            badbin_oyc[:,c] = shift(badbin_orc[:,c].astype(int),-drow_oc[c],cval=1,order=1) > bad_val
            var_oyc[:,c] = shift(var_orc[:,c],-drow_oc[c],order=1)
            wav_oyc[:,c] = shift(wav_orc[:,c],-drow_oc[c],order=1)

        #Create new masks that eliminate updated bad pixel mask after shifting
        okprof_oyc = ~badbin_oyc & okprof_rc
        okprofsm_oyc = ~badbin_oyc & okprofsm_rc
        
        
        #Sets wavelength values not on the spatial edges to 0 if the row
        #(spatial direction) before or after was 0 (posisbly from being outside boundary of spline above).
        # square off edge        
        wav_oyc[1:rows-1,:] *= np.logical_not((wav_oyc[1:rows-1,:]>0) & (wav_oyc[:rows-2,:]*wav_oyc[2:rows,:]==0)).astype(int)               

        #Compute median over sample of columns (wavelengths) to form profile as function of row (spatial direction)
        #Note axis =-1 selects the last axis dim in array
        profile_oy = np.median(profilesm_oyc,axis=-1)
        
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
        okprof_oyc[badrow_oy,:] = False            
        badbinnew_oyc[badrow_oy,:] = True

       
       
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
        #plot_profile(np.arange(Rows), profile_Y, mask=okprof_Y)

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
        
        #Mark only values with positive wavelength as OK
        okprof_oyc &= (wav_oyc > 0.)

        #Mark only values with valid values as OK
        okprof_oyc &= ~np.isnan(wav_oyc)

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

        #Set good pixels to True denoting background continuum?
        isbkgcont_oyc = ~(badbinnew_oyc)

        #Initialize new arrays
        targetrow_od = np.zeros((2))   
        badbinnew_orc = np.zeros_like(badbin_orc)
        isbkgcont_orc = np.zeros_like(badbin_orc)
        psf_orc = np.zeros_like(profile_orc)

        #Define parameters of 2D smoothing
        rblk = 1
        cblk = int(cols/16)

        #Smooth profile using defined block binning
        profile_oyc = blksmooth2d(profile_oyc,okprof_oyc,rblk,cblk,blklim=0.25,mode='mean')

        bad_val = 0.1
        for c in range(cols):
            #Move 2d smoothed profile back to initial position and save as PSF
            psf_orc[:,c] = shift(profile_oyc[:,c],drow_oc[c],cval=0,order=1)
            #Shift masks as well, flagging as bad any pixels with interpolated values > bad_val
            isbkgcont_orc[:,c] = shift(isbkgcont_oyc[:,c].astype(int),drow_oc[c],cval=0,order=1) > bad_val
            badbinnew_orc[:,c] = shift(badbinnew_oyc[:,c].astype(int),drow_oc[c],cval=1,order=1) > bad_val

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
        
        if debug:
            pf.PrimaryHDU(psf_orc.astype('float32')).writeto(sciname+"_psf_orc.fits",clobber=True) 
            #pf.PrimaryHDU(skyflat_orc.astype('float32')).writeto(sciname+"_skyflat_orc.fits",clobber=True)
            pf.PrimaryHDU(badbinnew_orc.astype('uint8')).writeto(sciname+"_badbinnew_orc.fits",clobber=True) 
            pf.PrimaryHDU(isbkgcont_orc.astype('uint8')).writeto(sciname+"_isbkgcont_orc.fits",clobber=True)           
        return psf_orc,badbinnew_orc,isbkgcont_orc,maprow_od,drow_oc,sciname,wav_orc


def specpolextract(image, hdu, obsname, wav_orc, maprow_od, drow_oc, badbinnew_orc,isbkgcont_orc):
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
    #Hmmm should we normalize by only the good pixels?
    psf_orw /= psf_orw.sum(axis=0)#[None,:]
       
    #plot_2d_profile(psf_orw,psf_orw.shape)   

    #Mask as zero the bad binned pixels
    target_orc = hdu['SCI'].data
    badbin_orc = (hdu['BPM'].data > 0)
    badbinbkg_orc = (badbin_orc | badbinnew_orc | isedge_orc | istarget_orc)
    target_orc *= (~badbin_orc).astype(int)

    #Get rid of NaNs
    target_orc[np.isnan(target_orc)] = 0.0

    if debug:
        pf.PrimaryHDU(target_orc.astype('float32')).writeto('target_orc.fits',clobber=True)

    #Store variance and bad pixel data
    var_orc = hdu['var'].data
    badbin_orc |= badbinnew_orc

    # extract spectrum optimally (Horne, PASP 1986)
    target_orw = np.zeros((rows,wavs))   
    var_orw = np.zeros_like(target_orw)
    badbin_orw = np.ones((rows,wavs),dtype='bool')   
    wt_orw = np.zeros_like(target_orw)

    #Identically scrunch the target, variance, and bad bin data
    bad_val = 0.001
    for r in specrow_or:
        target_orw[r] = scrunch1d(target_orc[r],binedge_orw[r])
        var_orw[r] = scrunch1d(var_orc[r],binedge_orw[r])
        #Flag any pixels that have interpolated BPM values > bad_val as bad
        badbin_orw[r] = scrunch1d(badbin_orc[r].astype(float),binedge_orw[r]) > bad_val

    #Add bins with variance at zero to bad pixel mask
    #??? Why would variance be zero after scrunching?
    badbin_orw |= (var_orw == 0)
    #Add rows with less than psfnormin flux to bad pixel mask
    badbin_orw |= ((psf_orw*(~badbin_orw)).sum(axis=0)[None,:] < psfnormmin)
    #NOTE: I get normalization > 1 and at 0 after using this mask for some rows.

    
    # use master psf shifted in row to allow for guide errors
    #pwidth = 2*int(1./psf_orw.max())
    #ok_w = ((psf_orw*badbin_orw).sum(axis=0) < 0.03/float(pwidth/2)).all(axis=0)
    #crosscor_s = np.zeros(pwidth)

    #for s in range(pwidth):
    #    crosscor_s[s] = (psf_orw[s:s-pwidth]*target_orw[pwidth/2:-pwidth/2]*ok_w).sum()

    #smax = np.argmax(crosscor_s)
    #s_S = np.arange(smax-pwidth/4,smax-pwidth/4+pwidth/2+1)
    
    #polycof = la.lstsq(np.vstack((s_S**2,s_S,np.ones_like(s_S))).T,crosscor_s[s_S])[0]
    #pshift = -(-0.5*polycof[1]/polycof[0] - pwidth/2) if (polycof[1] != 0.0 and polycof[0] != 0.0) else 0.0
    
    #s = int(pshift+pwidth)-pwidth
    #Ignore shifting the PSF for extraction...for now
    pshift = 0
    s = 0
    sfrac = pshift-s
    psfsh_orw = np.zeros_like(psf_orw)
    outrow = np.arange(max(0,s+1),rows-(1+int(abs(pshift)))+max(0,s+1))
    psfsh_orw[outrow] = (1.-sfrac)*psf_orw[outrow-s] + sfrac*psf_orw[outrow-s-1]

    #Define weights as inverse variance a la Horne
    wt_orw[~badbin_orw] = psfsh_orw[~badbin_orw]/var_orw[~badbin_orw]

    #Estimate variance in extracted spectrum
    var_ow = (psfsh_orw*wt_orw*(~badbin_orw)).sum(axis=0)
    badbin_ow = (var_ow == 0)
    var_ow[~badbin_ow] = 1./var_ow[~badbin_ow]
    
    sci_ow = (target_orw*wt_orw).sum(axis=0)*var_ow
    
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

    if len(sys.argv) != 2:
        print "Incorrect number of objects. Please supply image."
        sys.exit(1)


    image = sys.argv[1]
    hdu = pf.open(image)
    psf_orc,badbinnew_orc,isbkgcont_orc,maprow_od,drow_oc,sciname,wav_orc=specpolsignalmap(hdu,logfile="log.file",debug=False)
    specpolextract(image,hdu,sciname, wav_orc, maprow_od, drow_oc, badbinnew_orc,isbkgcont_orc=isbkgcont_orc)
