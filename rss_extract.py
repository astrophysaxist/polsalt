import os, sys, glob, shutil, inspect

import numpy as np
import pyfits
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from scipy.ndimage import convolve1d
from scipy import linalg as la

import reddir
datadir = os.path.dirname(inspect.getfile(reddir))+"/data/"

from oksmooth import boxsmooth1d,blksmooth2d
from pyraf import iraf
from iraf import pysalt
from saltobslog import obslog
from saltsafelog import logging

# np.seterr(invalid='raise')
np.set_printoptions(threshold=np.nan)
debug = True


def specpolsignalmap(hdu,logfile=sys.stdout,debug=False):
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
        sci_orc = hdu['sci'].data.copy()
        var_orc = hdu['var'].data.copy()
        badbin_orc = (hdu['bpm'].data > 0)
        wav_orc = hdu['wav'].data.copy()

        #Who are these guys?
        lam_m = np.loadtxt(datadir+"wollaston.txt",dtype=float,usecols=(0,))
        rpix_om = np.loadtxt(datadir+"wollaston.txt",dtype=float,unpack=True,usecols=(1,2))

        # trace spectrum, compute spatial profile 
        rows,cols = sci_orc.shape[1:3]
        sciname = hdu[0].header["OBJECT"]
        cbin,rbin = np.array(hdu[0].header["CCDSUM"].split(" ")).astype(int)
        slitid = hdu[0].header["MASKID"]
        if slitid[0] =="P": slitwidth = float(slitid[2:5])/10.
        else: slitwidth = float(slitid)         # in arcsec
        profsmoothfac = 2.5                     # profile smoothed by slitwidth*profsmoothfac

        lam_c = wav_orc[0,rows/2] #Why only half the row?
        profile_orc = np.zeros_like(sci_orc)
        profilesm_orc = np.zeros_like(sci_orc)
        drow_oc = np.zeros((2,cols))
        expectrow_oc = np.zeros((2,cols),dtype='float32')
        maxrow_oc = np.zeros((2,cols),dtype=int)
        maxval_oc = np.zeros((2,cols),dtype='float32')
        col_cr,row_cr = np.indices(sci_orc[0].T.shape)
        cross_or = np.sum(sci_orc[:,:,cols/2-cols/16:cols/2+cols/16],axis=2) #Summed over columns--why?
        
        okprof_oyc = np.ones((2,rows,cols),dtype='bool')
        okprofsm_oyc = np.ones((2,rows,cols),dtype='bool')
        profile_oyc = np.zeros_like(profile_orc)
        profilesm_oyc = np.zeros_like(profile_orc)
        badbin_oyc = np.zeros_like(badbin_orc)
        var_oyc = np.zeros_like(var_orc)
        wav_oyc = np.zeros_like(wav_orc)
        badbinnew_oyc = np.zeros_like(badbin_orc)

        for o in (0,1):
        # find spectrum roughly from max of central cut, then within narrow curved aperture
            expectrow_oc[o] = (1-o)*rows + interp1d(lam_m,rpix_om[o],kind='cubic',bounds_error=False)(lam_c)/rbin    
            okwav_c = ~np.isnan(expectrow_oc[o])
            goodcols = okwav_c.sum()
            crossmaxval = np.max(cross_or[o,expectrow_oc[o,cols/2]-100/rbin:expectrow_oc[o,cols/2]+100/rbin])
            drow = np.where(cross_or[o]==crossmaxval)[0][0] - expectrow_oc[o,cols/2]
            row_c = (expectrow_oc[o] + drow).astype(int)
#            if debug: np.savetxt(sciname+"_row_c_"+str(o)+".txt",row_c,fmt="%8.3f")
            isaper_cr = ((row_cr-row_c[:,None])>=-20/rbin) & ((row_cr-row_c[:,None])<=20/rbin) & okwav_c[:,None]       
            maxrow_oc[o,okwav_c] = np.argmax(sci_orc[o].T[isaper_cr].reshape((goodcols,-1)),axis=1) \
                                    + row_c[okwav_c] - 20/rbin
            maxval_oc[o,okwav_c] = sci_orc[o,maxrow_oc[o,okwav_c],okwav_c]
            drow1_c = maxrow_oc[o] -(expectrow_oc[o] + drow)
            trow_o = maxrow_oc[:,cols/2]
#            if debug:
#                np.savetxt(sciname+"_drow1_c_"+str(o)+".txt",drow1_c,fmt="%8.3f")
#                np.savetxt(sciname+"_maxrow_c_"+str(o)+".txt",maxrow_oc[o],fmt="%8.3f")
#                np.savetxt(sciname+"_maxval_c_"+str(o)+".txt",maxval_oc[o],fmt="%8.3f")
#                np.savetxt(sciname+"_okwav_c_"+str(o)+".txt",okwav_c,fmt="%4i")

        # divide out spectrum (allowing for spectral curvature) to make spatial profile
            okprof_c = (maxval_oc[o] != 0) & okwav_c
            drow2_c = np.polyval(np.polyfit(np.where(okprof_c)[0],drow1_c[okprof_c],3),(range(cols)))
#            if debug: np.savetxt(sciname+"_drow2_c_"+str(o)+".txt",drow2_c,fmt="%8.3f")
            okprof_c[okwav_c] &= np.abs(drow2_c - drow1_c)[okwav_c] < 3
            norm_rc = np.zeros((rows,cols));    normsm_rc = np.zeros((rows,cols))
            for r in range(rows):
                norm_rc[r] = interp1d(wav_orc[o,trow_o[o],okprof_c],maxval_oc[o,okprof_c], \
                    bounds_error = False , fill_value=0.)(wav_orc[o,r])
            okprof_rc = (norm_rc != 0.)

        # make a slitwidth smoothed norm and profile for the background area
            for r in range(rows):
                normsm_rc[r] = boxsmooth1d(norm_rc[r],okprof_rc[r],(8.*slitwidth*profsmoothfac)/cbin,0.5)
            okprofsm_rc = (normsm_rc != 0.)
            profile_orc[o,okprof_rc] = sci_orc[o,okprof_rc]/norm_rc[okprof_rc]
            profilesm_orc[o,okprofsm_rc] = sci_orc[o,okprofsm_rc]/normsm_rc[okprofsm_rc]
            var_orc[o,okprof_rc] = var_orc[o,okprof_rc]/norm_rc[okprof_rc]**2
            drow_oc[o] = (expectrow_oc[o] - expectrow_oc[o,cols/2] + drow2_c -drow2_c[cols/2])

        # take out profile spatial curvature and tilt (r -> y)
            for c in range(cols):
                profile_oyc[o,:,c] = shift(profile_orc[o,:,c],-drow_oc[o,c],order=1)
                profilesm_oyc[o,:,c] = shift(profilesm_orc[o,:,c],-drow_oc[o,c],order=1)
                badbin_oyc[o,:,c] = shift(badbin_orc[o,:,c].astype(int),-drow_oc[o,c],cval=1,order=1) > 0.1
                var_oyc[o,:,c] = shift(var_orc[o,:,c],-drow_oc[o,c],order=1)
                wav_oyc[o,:,c] = shift(wav_orc[o,:,c],-drow_oc[o,c],order=1)
            okprof_oyc[o] = ~badbin_oyc[o] & okprof_rc
            okprofsm_oyc[o] = ~badbin_oyc[o] & okprofsm_rc
        wav_oyc[:,1:rows-1] *= np.logical_not((wav_oyc[:,1:rows-1]>0) & \
                (wav_oyc[:,:rows-2]*wav_oyc[:,2:rows]==0)).astype(int)                  # square off edge
        profile_oy = np.median(profilesm_oyc,axis=-1)
#        if debug:
#            pyfits.PrimaryHDU(norm_rc.astype('float32')).writeto(sciname+'_norm_rc.fits',clobber=True)
#            pyfits.PrimaryHDU(normsm_rc.astype('float32')).writeto(sciname+'_normsm_rc.fits',clobber=True)
#            pyfits.PrimaryHDU(profile_orc.astype('float32')).writeto(sciname+'_profile_orc.fits',clobber=True)
#            pyfits.PrimaryHDU(profilesm_orc.astype('float32')).writeto(sciname+'_profilesm_orc.fits',clobber=True)
#            pyfits.PrimaryHDU(profilesm_oyc.astype('float32')).writeto(sciname+'_profilesm_oyc.fits',clobber=True)
#            pyfits.PrimaryHDU(wav_oyc.astype('float32')).writeto(sciname+'_wav_oyc.fits',clobber=True)  
#            np.savetxt(sciname+"_drow_oc.txt",drow_oc.T,fmt="%8.3f %8.3f")
#            np.savetxt(sciname+'_profile_oy.txt',profile_oy.T,fmt="%10.5f")

    # Take FOV from wavmap
        edgerow_od = np.zeros((2,2))
        badrow_oy = np.zeros((2,rows),dtype=bool)
        for o in (0,1):
            edgerow_od[o,0] = np.argmax(wav_oyc[o,:,cols/2] > 0.)
            edgerow_od[o,1] = rows-np.argmax((wav_oyc[o,:,cols/2] > 0.)[::-1])
            badrow_oy[o] = ((np.arange(rows)<edgerow_od[o,0]) | (np.arange(rows)>edgerow_od[o,1]))
        axisrow_o = edgerow_od.mean(axis=1)

        okprof_oyc[badrow_oy,:] = False            
        badbinnew_oyc[badrow_oy,:] = True 

        log.message('Optical axis row:  O    %4i     E    %4i' % tuple(axisrow_o), with_header=False)
        log.message('Target center row: O    %4i     E    %4i' % tuple(trow_o), with_header=False)
        log.message('Bottom, top row:   O %4i %4i   E %4i %4i \n' \
                % tuple(edgerow_od.flatten()), with_header=False)
                
    # Mask off atmospheric A- and B-band (profile only) 
        ABband = np.array([[7582.,7667.],[6856.,6892.]])         
        for b in (0,1): okprof_oyc &= ~((wav_oyc>ABband[b,0])&(wav_oyc<ABband[b,1]))

        profile_oyc *= okprof_oyc
        profilesm_oyc *= okprofsm_oyc

    # Stray light search by looking for seeing-sized features in spatial and spectral profile                   
        profile_Y = 0.5*(profile_oy[0,trow_o[0]-16:trow_o[0]+17] + \
                        profile_oy[1,trow_o[1]-16:trow_o[1]+17])
        profile_Y = interp1d(np.arange(-16.,17.),profile_Y,kind='cubic')(np.arange(-16.,16.,1./16))
        fwhm = 3.*(np.argmax(profile_Y[256:]<0.5) + np.argmax(profile_Y[256:0:-1]<0.5))/16.
        kernelcenter = np.ones(np.around(fwhm/2)*2+2)
        kernelbkg = np.ones(kernelcenter.shape[0]+4)
        kernel = -kernelbkg*kernelcenter.sum()/(kernelbkg.sum()-kernelcenter.sum())
        kernel[2:-2] = kernelcenter                # ghost search kernel is size of 3*fwhm and sums to zero

    # First, look for second order as feature in spatial direction
        ghost_oyc = convolve1d(profilesm_oyc,kernel,axis=1,mode='constant',cval=0.)
        isbadghost_oyc = (~okprofsm_oyc | badbin_oyc | badbinnew_oyc)     
        isbadghost_oyc |= convolve1d(isbadghost_oyc.astype(int),kernelbkg,axis=1,mode='constant',cval=1) != 0
        for o in(0,1): isbadghost_oyc[o,trow_o[o]-3*fwhm/2:trow_o[o]+3*fwhm/2,:] = True
        ghost_oyc *= (~isbadghost_oyc).astype(int)                
        stdghost_oc = np.std(ghost_oyc,axis=1)
        boxbins = (int(2.5*fwhm)/2)*2 + 1
        boxrange = np.arange(-int(boxbins/2),int(boxbins/2)+1)

    # _C: columns where second order is possible
        col_C = np.arange(np.argmax(lam_c/2. > lam_m[0]),np.where(okwav_c)[0].max())  
        is2nd_oyc = np.zeros((2,rows,cols),dtype=bool)
        row2nd_oYC = np.zeros((2,boxbins,col_C.shape[0]),dtype=int)

        for o in (0,1):
            row2nd_C = np.around(trow_o[o]  \
                + (interp1d(lam_m,rpix_om[o],kind='cubic',bounds_error=False)(lam_c[col_C]/2.)  \
                - interp1d(lam_m,rpix_om[o],kind='cubic',bounds_error=False)(lam_c[col_C]))/rbin).astype(int)
            row2nd_oYC[o] = row2nd_C + boxrange[:,None]
            is2nd_oyc[o][row2nd_oYC[o],col_C] = ghost_oyc[o][row2nd_oYC[o],col_C] > 3.*stdghost_oc[o,col_C]
            
    # Mask off second order (profile and image), using box found above on profile
        is2nd_c = np.any(np.all(is2nd_oyc,axis=0),axis=0)
#        pyfits.PrimaryHDU(okprof_oyc.astype('uint8')).writeto(sciname+'_okprof_oyc.fits',clobber=True) 
        if is2nd_c.sum() > 100: 
            is2nd_c = np.any(np.all(is2nd_oyc,axis=0),axis=0)
            col2nd1 = np.where(is2nd_c)[0][-1]
            col2nd0 = col2nd1 - np.where(~is2nd_c[col2nd1::-1])[0][0] +1
            col_C = np.arange(col2nd0,col2nd1+1)
            row2nd_oYC = np.zeros((2,boxbins,col_C.shape[0]),dtype=int)
            is2nd_oyc = np.zeros((2,rows,cols),dtype=bool)        
            for o in (0,1):
                row2nd_C = np.around(trow_o[o] + (interp1d(lam_m,rpix_om[o],kind='cubic')(lam_c[col_C]/2.)  \
                    - interp1d(lam_m,rpix_om[o],kind='cubic')(lam_c[col_C]))/rbin).astype(int)
                row2nd_oYC[o] = row2nd_C + boxrange[:,None]
                for y in np.arange(edgerow_od[o,0],edgerow_od[o,1]+1):
                    profile_oy[o,y] = np.median(profilesm_oyc[o,y,okprof_oyc[o,y,:]])
                dprofile_yc = profilesm_oyc[o] - profile_oy[o,:,None]
                is2nd_oyc[o][row2nd_oYC[o],col_C] = \
                    (dprofile_yc[row2nd_oYC[o],col_C] > 5.*np.sqrt(var_oyc[o][row2nd_oYC[o],col_C]))
                strength2nd = dprofile_yc[is2nd_oyc[o]].max()
            is2nd_c = np.any(np.all(is2nd_oyc,axis=0),axis=0)
            col2nd1 = np.where(is2nd_c)[0][-1]
            col2nd0 = col2nd1 - np.where(~is2nd_c[col2nd1::-1])[0][0] +1
            is2nd_oyc[:,:,0:col2nd0-1] = False
            wav2nd0,wav2nd1 = wav_oyc[0,trow_o[0],[col2nd0,col2nd1]]
            okprof_oyc &= (~is2nd_oyc)
            okprofsm_oyc &= (~is2nd_oyc)
            badbinnew_oyc |= is2nd_oyc
            isbadghost_oyc |= is2nd_oyc
            ghost_oyc *= (~isbadghost_oyc).astype(int)                     
            log.message('2nd order masked,     strength %7.4f, wavel %7.1f - %7.1f (/2)' \
                        % (strength2nd,wav2nd0,wav2nd1), with_header=False)
        else: col2nd0 = cols        

#        np.savetxt(sciname+"_stdghost_oc.txt",stdghost_oc.T,fmt="%10.5f")
#        pyfits.PrimaryHDU(isbadghost_oyc.astype('uint8')).writeto(sciname+'_isbadghost_oyc.fits',clobber=True) 
#        pyfits.PrimaryHDU(ghost_oyc.astype('float32')).writeto(sciname+'_ghost_oyc.fits',clobber=True) 


    # Remaining ghosts have same position O and E. _Y = row around target, both beams
        Rows = int(2*np.abs(trow_o[:,None]-edgerow_od).min()+1)
        row_oY = np.add.outer(trow_o,np.arange(Rows)-(Rows+1)/2).astype(int)
        ghost_Yc = 0.5*ghost_oyc[np.arange(2)[:,None],row_oY,:].sum(axis=0) 
        isbadghost_Yc = isbadghost_oyc[np.arange(2)[:,None],row_oY,:].any(axis=0)
        stdghost_c = np.std(ghost_Yc,axis=0)
        profile_Yc = 0.5*profilesm_oyc[np.arange(2)[:,None],row_oY,:].sum(axis=0) 
        okprof_Yc = okprof_oyc[np.arange(2)[:,None],row_oY,:].all(axis=0)               
                    
    # Search for Littrow ghost as undispersed object off target
    # Convolve with ghost kernal in spectral direction, divide by standard deviation, 
    #  then add up those > 10 sigma within fwhm box
        isbadlitt_Yc = isbadghost_Yc | \
            (convolve1d(isbadghost_Yc.astype(int),kernelbkg,axis=1,mode='constant',cval=1) != 0)
        litt_Yc = convolve1d(ghost_Yc,kernel,axis=-1,mode='constant',cval=0.)*(~isbadlitt_Yc).astype(int)
        litt_Yc[:,stdghost_c>0] /= stdghost_c[stdghost_c>0]
        litt_Yc[litt_Yc < 10.] = 0.
        for c in range(cols):
            litt_Yc[:,c] = np.convolve(litt_Yc[:,c],np.ones(boxbins))[boxbins/2:boxbins/2+Rows] 
        for Y in range(Rows):
            litt_Yc[Y] = np.convolve(litt_Yc[Y],np.ones(boxbins))[boxbins/2:boxbins/2+cols] 
        Rowlitt,collitt = np.argwhere(litt_Yc == litt_Yc[:col2nd0].max())[0]
        littbox_Yc = np.meshgrid(boxrange+Rowlitt,boxrange+collitt)

#        if debug: 
#            np.savetxt(sciname+"_stdghost_c.txt",stdghost_c.T,fmt="%10.5f")
#            pyfits.PrimaryHDU(litt_Yc.astype('float32')).writeto(sciname+'_litt_Yc.fits',clobber=True)
#            pyfits.PrimaryHDU(isbadlitt_Yc.astype('uint8')).writeto(sciname+'_isbadlitt_Yc.fits',clobber=True)
  
    # Mask off Littrow ghost (profile and image)
        if litt_Yc[Rowlitt,collitt] > 100:
            islitt_oyc = np.zeros((2,rows,cols),dtype=bool)
            for o in (0,1):
                for y in np.arange(edgerow_od[o,0],edgerow_od[o,1]+1):
                    if profilesm_oyc[o,y,okprof_oyc[o,y,:]].shape[0] == 0: continue
                    profile_oy[o,y] = np.median(profilesm_oyc[o,y,okprof_oyc[o,y]])
                dprofile_yc = profilesm_oyc[o] - profile_oy[o,:,None]
                littbox_yc = np.meshgrid(boxrange+Rowlitt-Rows/2+trow_o[o],boxrange+collitt)
                islitt_oyc[o][littbox_yc] =  \
                    dprofile_yc[littbox_yc] > 10.*np.sqrt(var_oyc[o][littbox_yc])
            if islitt_oyc.sum():
                wavlitt = wav_oyc[0,trow_o[0],collitt]
                strengthlitt = dprofile_yc[littbox_yc].max()
                okprof_oyc[islitt_oyc] = False
                badbinnew_oyc |= islitt_oyc
                isbadghost_Yc[littbox_Yc] = True
                ghost_Yc[littbox_Yc] = 0.
            
                log.message('Littrow ghost masked, strength %7.4f, ypos %5.1f", wavel %7.1f' \
                        % (strengthlitt,(Rowlitt-Rows/2)*(rbin/8.),wavlitt), with_header=False)        

    # Anything left as spatial profile feature is assumed to be neighbor non-target stellar spectrum
    # Mask off spectra above a threshhold
        okprof_Yc = okprof_oyc[np.arange(2)[:,None],row_oY,:].all(axis=0)
        okprof_Y = okprof_Yc.any(axis=1)
        profile_Y = np.zeros(Rows,dtype='float32')
        for Y in np.where(okprof_Y)[0]: profile_Y[Y] = np.median(profile_Yc[Y,okprof_Yc[Y]])
        avoid = int(np.around(fwhm/2)) +5
        okprof_Y[range(avoid) + range(Rows/2-avoid,Rows/2+avoid) + range(Rows-avoid,Rows)] = False
        nbr_Y = convolve1d(profile_Y,kernel,mode='constant',cval=0.)
        nbrmask = 1.0
        count = 0
        while (nbr_Y[okprof_Y]/profile_Y[okprof_Y]).max() > nbrmask:
            count += 1
            nbrrat_Y = np.zeros(Rows)
            nbrrat_Y[okprof_Y] = nbr_Y[okprof_Y]/profile_Y[okprof_Y]
            Ynbr = np.where(nbrrat_Y == nbrrat_Y.max())[0]
            Ymask1 = Ynbr - np.argmax(nbrrat_Y[Ynbr::-1] < nbrmask)
            Ymask2 = Ynbr + np.argmax(nbrrat_Y[Ynbr:] < nbrmask)
            strengthnbr = nbr_Y[Ynbr]/nbr_Y[Rows/2]
            okprof_Y[Ymask1:Ymask2+1] = False
            for o in (0,1):
                badbinnew_oyc[o,row_oY[o,Ymask1:Ymask2],:] = True 
                okprof_oyc[o,row_oY[o,Ymask1:Ymask2],:] = False

            log.message('Neighbor spectrum masked: strength %7.4f, ypos %5.1f' \
                            % (strengthnbr,(Ynbr-Rows/2)*(rbin/8.)), with_header=False)
            if count>10: break

        if debug: np.savetxt(sciname+"_nbrdata_Y.txt",np.vstack((profile_Y,nbr_Y,okprof_Y.astype(int))).T,fmt="%8.5f %8.5f %3i")         

        okprof_oyc &= (wav_oyc > 0.)
        hduprof = pyfits.PrimaryHDU(header=hdu[0].header)   
        hduprof = pyfits.HDUList(hduprof)  
        header=hdu['SCI'].header.copy()       
        hduprof.append(pyfits.ImageHDU(data=profilesm_oyc.astype('float32'), header=header, name='SCI'))
        hduprof.append(pyfits.ImageHDU(data=(var_oyc*okprof_oyc).astype('float32'), header=header, name='VAR'))
        hduprof.append(pyfits.ImageHDU(data=(~okprof_oyc).astype('uint8'), header=header, name='BPM'))
        hduprof.append(pyfits.ImageHDU(data=wav_oyc.astype('float32'), header=header, name='WAV'))
        if debug: hduprof.writeto(sciname+"_skyflatprof.fits",clobber=True)
        corerows = (profile_Y - np.median(profile_Y[profile_Y>0]) > 0.015).sum()

        lsqcof_oC,bkg_oyc,badlinerow_oy,badbinmore_oyc,isline_oyc = \
            skyflat(hduprof,trow_o,corerows,axisrow_o,log,datadir,debug)

    # compute skyflat in original geometry
        skyflat_orc = np.ones((2,rows,cols))
        Cofs = lsqcof_oC.shape[0]
        if Cofs>0:
            wavmin,wavmax = np.around([wav_oyc[:,rows/2,:].min(),wav_oyc[:,rows/2,:].max()],1)
            for o in (0,1):
                outr_f,outc_f = np.indices((rows,cols)).reshape((2,-1))
                drow_f = np.broadcast_arrays(-drow_oc[o],np.zeros((rows,cols)))[0].flatten()
                outrow_f = (outr_f - axisrow_o[o] + drow_f)/(0.5*rows)
                outwav_f = (wav_orc[o].flatten()-np.mean([wavmin,wavmax]))/(0.5*(wavmax - wavmin))                                
                aout_fC = (np.vstack((outrow_f,outrow_f**2,outrow_f*outwav_f,outrow_f**2*outwav_f))).T             
                skyflat_orc[o] = (np.dot(aout_fC,lsqcof_oC[o]) + 1.).reshape((rows,cols))

    # compute stellar psf in original geometry for extraction
    # use profile normed to (unsmoothed) stellar spectrum, new badpixmap, removing background continuum 
        badbinnew_oyc |= badbinmore_oyc                
        isbkgcont_oyc = ~(badbinnew_oyc | isline_oyc | badlinerow_oy[:,:,None])
        targetrow_od = np.zeros((2,2))   
        badbinnew_orc = np.zeros_like(badbin_orc)
        isbkgcont_orc = np.zeros_like(badbin_orc)
        psf_orc = np.zeros_like(profile_orc)      
        isedge_oyc = (np.arange(rows)[:,None] < edgerow_od[:,None,None,0]) | \
            (np.indices((rows,cols))[0] > edgerow_od[:,None,None,1])
        isskycont_oyc = (((np.arange(rows)[:,None] < edgerow_od[:,None,None,0]+rows/16) |  \
            (np.indices((rows,cols))[0] > edgerow_od[:,None,None,1]-rows/16)) & ~isedge_oyc)

        for o in (0,1):                         # yes, it's not quite right to use skyflat_o*r*c                      
            skycont_c = (bkg_oyc[o].T[isskycont_oyc[o].T]/ \
                skyflat_orc[o].T[isskycont_oyc[o].T]).reshape((cols,-1)).mean(axis=-1)
            skycont_yc = skycont_c*skyflat_orc[o]           
            profile_oyc[o] -= skycont_yc
            rblk = 1; cblk = int(cols/16)
            
            profile_oyc[o] = blksmooth2d(profile_oyc[o],(okprof_oyc[o] & ~isline_oyc[o]),   \
                        rblk,cblk,0.25,mode='mean')              
            for c in range(cols):
                psf_orc[o,:,c] = shift(profile_oyc[o,:,c],drow_oc[o,c],cval=0,order=1)
                isbkgcont_orc[o,:,c] = shift(isbkgcont_oyc[o,:,c].astype(int),drow_oc[o,c],cval=0,order=1) > 0.1
                badbinnew_orc[o,:,c] = shift(badbinnew_oyc[o,:,c].astype(int),drow_oc[o,c],cval=1,order=1) > 0.1
            targetrow_od[o,0] = trow_o[o] - np.argmax(isbkgcont_orc[o,trow_o[o]::-1,cols/2] > 0)
            targetrow_od[o,1] = trow_o[o] + np.argmax(isbkgcont_orc[o,trow_o[o]:,cols/2] > 0)

        maprow_od = np.vstack((edgerow_od[:,0],targetrow_od[:,0],targetrow_od[:,1],edgerow_od[:,1])).T
        maprow_od += np.array([-2,-2,2,2])

        if debug:
            pyfits.PrimaryHDU(psf_orc.astype('float32')).writeto(sciname+"_psf_orc.fits",clobber=True) 
            pyfits.PrimaryHDU(skyflat_orc.astype('float32')).writeto(sciname+"_skyflat_orc.fits",clobber=True)
            pyfits.PrimaryHDU(badbinnew_orc.astype('uint8')).writeto(sciname+"_badbinnew_orc.fits",clobber=True) 
            pyfits.PrimaryHDU(isbkgcont_orc.astype('uint8')).writeto(sciname+"_isbkgcont_orc.fits",clobber=True)           
        return psf_orc,skyflat_orc,badbinnew_orc,isbkgcont_orc,maprow_od,drow_oc
                        
