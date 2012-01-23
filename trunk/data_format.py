#!/usr/bin/env python
#
# Name:  data format 
#
# Author: Thuso S Simon
#
# Date: 20/1/12
# TODO: add citations
#
#    vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#    Copyright (C) 2011  Thuso S Simon
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY# without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    For the GNU General Public License, see <http://www.gnu.org/licenses/>.
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#History (version,date, change author)
#
#
#
'''
does all fits file stuff to format correctly for AGE_date moduals
most code is taken from ULYSS will site later
'''

import numpy as nu
import pyfits as fits


def spect_read(infile):
# Analyse the header of a FITS file to guess what format it is
# The routine is presently recognizing 3 spectral formats.
# The aim is perform minimum tests in order to select the proper reading 
# algorithm. It does not warrant that the file is actually readable.
#
# The function returns:
#   0 : The format is not recognized
#   1 : SDSS-like format
#   2 : Spectrum in columns of a BINTABLE
#   3 : 1D or LSS-2D spectrum with a 'fair' WCS
# If the variable hdr does not contain a FITS header, the header can 
# be read from <file>
    hdr=fits.open(infile)

# Search if it is a SDSS-like format (ie. have ARRAYi)
    if hdr[0].header['NAXIS']>1:  
        array = hdr[0].header['ARRAY*']
        naxis = hdr[0].header['NAXIS*']
        out_spect=spect_read_sdss(hdr,array,naxis)
'''
# Search if it is a BINTABLE format 
if strtrim(sxpar(hdr, 'XTENSION')) eq 'BINTABLE' then begin  
    return, 2
endif

# Search if it can be a 1D or LSS spectrum in image array
if sxpar(hdr, 'NAXIS') le 3 then begin  
    return, 3
endif

return, 0
end
'''
#==============================================================================
# reading routine for SDSS-style format (format=1)
def spect_read_sdss(hdr,array,naxis, quiet=True):

    if not QUIET:
        print 'SDSS style'

# pb with sdss style: the definition of MASK in SDSS original data
# is complex (this is a bit mask, 0 is good)
# We would like to understand also a simpler mask made of 0 and 1s, 1
# is good...

# identify the subarrays that we will extract
    n=-nu.ones(4,dtype=nu.int32)
    for i in range(len(array)):
        if array[i].value=='DATA' or array[i].value=='SPECTRUM':
            n[0]=i
        elif array[i].value=='ERROR':
            n[1]=i
        elif array[i].value=='MASK':
            n[2]=i
        elif array[i].value=='WAVELEN':
            n[3]=i


    if  hdr[0].header['CTYPE1'][:4]=='WAVE' or hdr[0].header['CTYPE1'][:4]=='AWAV':
        if hdr[0].header['CTYPE1'][5:]== 'WAV':
            sampling = 0
        elif hdr[0].header['CTYPE1'][5:]== 'LOG':
            sampling = 1
    try:
        step = hdr[0].header['CD1_1']
    except KeyError:
        step = hdr[0].header['CDELT1']
    except:
        print 'Cannot decode the WCS'
        raise(KeyError)
    
    crpix = hdr[0].header['CRPIX1']
    
    start = hdr[0].header['CRVAL1'] - crpix*step
    

    if sampling < 0:
        if start < 4 then begin
            sampling = 1 
            start = nu.log10(start)
            step = nu.log10(step)
            if not quiet :
                print 'Assume that the sampling is in log10'
        elif start < 9:
            sampling = 1
            if not quiet :
                print 'Assume that the sampling is in ln'
        else:
            sampling = 0
            if not quiet :
                print'Assume that the sampling is linear in wavelength'

    if vacuum == 1:
        if not quiet:
            print 'Wavelength in VACUUM ... approximately converted'
        if sampling == 0:
            start /= 1.00028
            step /= 1.00028
        elif sampling == 1:
             start -= 0.00028

        else :
            sampling = 2

# reformat the data array
ndim = size(data,/N_DIM)
dim = size(data, /DIM)

narray = dim[ndim-1]
dim = dim[0:ndim-2]
tot = 1
for i=0, ndim-2 do tot *= dim[i]
data = reform(data, tot, narray, /OVER)

spect = uly_spect_alloc(START=start, STEP=step, SAMP=sampling, HEADER=h)

# Remove some WCS keywords  (and ARRAY*) that may be outdated
sxdelpar, *spect.hdr, ['VACUUM', 'CTYPE1', 'CRVAL1', 'CDELT1', 'CD1_1', 'DC-FLAG', 'WAT0_*', 'WAT1_*', 'WFITTYPE', 'ARRAY*']

if n1 ge 0 then *spect.data = reform(data[*,n1], dim)
if n2 ge 0 then *spect.err = reform(data[*,n2], dim)

if n3 ge 0 then begin
    m = where(finite(data[*,n3]) eq 1, cnt)
    if cnt gt 0 then begin
        m = where(data[*,n3] ne 0 and data[*,n3] ne 1, cnt)
        if cnt gt 0 then begin
            message, /INFO, $
              'The mask is not made of 0 and 1s ... it is ignored' 
            message, /INFO, '   (a standard mask has 0=bad, 1=good)' 
        endif else $
          *spect.goodpix = where(data[*,n3] gt 0) #goodpix is a 1D list
    endif
endif

if n4 ge 0 then *spect.wavelen = reform(data[*,n4], dim)

if n_elements(*spect.err) gt 0 then begin
    if n_elements(*spect.goodpix) eq 0 then $
      m = where(*spect.err le 0, cnt, COMP=g) $
    else m = where((*spect.err)[*spect.goodpix] le 0, cnt, COMP=g)
    if cnt gt 0 then begin
        message, /INFO, 'Pixels with 0 or negative errors were masked'
        if n_elements(*spect.goodpix) eq 0 then *spect.goodpix = g $
        else *spect.goodpix = [*spect.goodpix, g]
    endif
endif

dof_factor = double(string(sxpar(h, 'DOF_FACT', COUNT=count)))
if count eq 1 then spect.dof_factor = dof_factor

return, spect

end

#==============================================================================
# reading routine for BINTABLE format (format=2)
function uly_spect_read_tbl, data, h, ERR_SP=err_sp, SNR_SP=snr_sp, QUIET=quiet
  
# we should there find what collumns to read (for data, error and mask)
  
# for the moment it is only suited to the spectra produced by spec2d,
# the pipeline of DEIMOS: 
# http://astro.berkeley.edu/~cooper/deep/spec2d/primer.html
  
# IVAR is the inverse of the variance of each px
if tag_exist(data, 'IVAR') then begin
   zeros = where(data.ivar eq 0, cnt, compl=goodpix)
   if cnt gt 0 then data.ivar[zeros] = 1 #we put arbitary value for the masked pxs
   err = 1d/sqrt(data.ivar)
endif

return, uly_spect_alloc(DATA=data.spec, WAVELEN=data.lambda, ERR=err, GOODPIX=goodpix, SAMP=2)

end

#==============================================================================
# reading routine of 1D and LSS spectra (format=3)
function uly_spect_read_lss, data, h, DISP_AXIS=disp_axis, QUIET=quiet

SignalLin = uly_spect_alloc(DATA=data, HEAD=h)
naxis = sxpar(h, 'NAXIS')


# ----------------------------------------------------------------------------
# search the dispersion axis, and the dispersion type (lin or log)
disp_type = -1
if n_elements(disp_axis) le 0 then disp_axis = 0
vacuum = 0  # Set to 1 if the wavelengths are in vacuum

if disp_axis le 0 then begin #  is there a standard WCS? (Vacuum wavelength)
    ctype = sxpar(h,'CTYPE*',COUNT=cnt)
    if cnt gt 0 then begin
        if ctype[0] eq 'WAVE-WAV' then begin #if it is 2d we need the first CTYPE
            creator = sxpar(h,'CREATOR', COUNT=count) # try to patch a Pleinpot bug
            if count gt 0 and creator eq 'Pleinpot     1' then ctype = 'AWAV'
#  in Elodie lib. 'WAVE' means air (in fits standarts 'AWAV' is air wavelength)
        endif
        disp_axis = 1 + where(strtrim(ctype,2) eq 'WAVE' or strmid(ctype,0,5) eq 'WAVE-') 
        if disp_axis gt 0 then begin
            if not keyword_set(quiet) then $
              print, 'Dispersion axis is ', strtrim(disp_axis[0],2), ' (Vacuum wavelength)'
            vacuum = 1
        endif
    endif
endif

if disp_axis le 0 then begin #  is there a standard WCS? (Air wavelength)
    disp_axis = 1 + where((strmid(ctype,0,4) eq 'AWAV') eq 1) 
    if disp_axis gt 0 then begin
        if not keyword_set(quiet) then $
          print, 'Dispersion axis is ', strtrim(disp_axis[0],2), ' (Air wavelength)'
        if max(strmid(ctype[disp_axis-1],5,3) eq ['   ', 'WAV']) then begin
            disp_type = 0
            if not keyword_set(quiet) then print, 'Dispersion axis is linear'
        endif else if strmid(ctype[disp_axis-1],5,3) eq 'LOG' then begin
            disp_type = 1
            if not keyword_set(quiet) then $
              print, 'Dispersion axis is logarithmic'
        endif
    endif
endif

if disp_axis eq 0 then begin
    disp_axis = 1
    if naxis gt 1 and not keyword_set(quiet) then $
      message, 'We assume that the dispersion axis is 1 (X)', /INFO
endif else $
  if disp_axis eq 2 then *SignalLin.data = transpose(*SignalLin.data)

ax = strtrim(disp_axis[0],2)
crval = double(string(sxpar(h,'CRVAL'+ax)))

if disp_type lt 0 then begin    # search the dispersion mode
    disp_type = 0
    if crval lt 10 then begin
        disp_type = 1
        if crval lt 5 then begin
            if not keyword_set(quiet) then $
              print, 'Assume that the dispersion is in log10',crval,'CRVAL'+ax
        endif else if not keyword_set(quiet) then $
          print, 'Assume that the dispersion is in log (air wavelength)'
    endif else if not keyword_set(quiet) then $
      print, 'Assume that the dispersion is linear (air wavelength)'
endif 

if disp_type ne 0 and disp_type ne 1 then $
  message, 'Do not handle this sampling (yet)'

# ----------------------------------------------------------------------------
# decode the spectral WCS
cdelt = double(string(sxpar(h,'CD'+ax+'_'+ax, COUNT=count)))
if count eq 0 then cdelt = double(string(sxpar(h,'CDELT'+ax)))

crpix = double(sxpar(h, 'CRPIX'+ax, COUNT=count))
if count eq 0 then crpix = 1d
crval = crval - (crpix - 1d) * cdelt # wavelength of the 1st pixel

if (cdelt le 0.) or (crval le 0.) then $
  message,'WCS of the observations not set correctly'

if disp_type eq 1 and crval lt 5 then begin  
#   normally axis should be logn, but sometime it is log 10 ...
    if not keyword_set(quiet) then $
      print, 'Convert axis scale from log10 to log'
    crval *= alog(10d)
    cdelt *= alog(10d)
endif

if vacuum eq 1 then begin
    if not keyword_set(quiet) then $
      print, 'Wavelength in VACUUM ... approximately converted'
    if disp_type eq 0 then begin 
        crval /= 1.00028d 
        cdelt /= 1.00028d 
    endif else if disp_type eq 1 then crval -= 0.00028D
endif

# ----------------------------------------------------------------------------
# load the output structure

SignalLin.sampling = disp_type # sampling in wavelength: lin/log
SignalLin.start = crval
SignalLin.step = cdelt

dof_factor = double(string(sxpar(h, 'DOF_FACT', COUNT=count)))
if count eq 1 then SignalLin.dof_factor = dof_factor

*SignalLin.hdr = h[3:*]      # initialize the header
sxdelpar, *SignalLin.hdr, $     # remove wcs and array specific keywords
  ['NAXIS1', 'CRVAL1', 'CRPIX1', 'CD1_1', 'CDELT1', 'CTYPE1', 'CROTA1', $
   'CD2_1', 'CD1_2', 'DATAMIN', 'DATAMAX', 'CHECKSUM' $
  ]

return, SignalLin

end

#==============================================================================
function uly_spect_read, file_in, lmin, lmax,                         $
                         VELSCALE=velscale, SG=sg,                    $
                         ERR_SP=err_sp, SNR_SP=snr_sp, MSK_SP=msk_sp, $
                         DISP_AXIS=disp_axis, QUIET=quiet

## read the first extension that we hope contains a spectrum
#fits_read, file_in, data, h

# test if the file or unit argument is valid
if size(file_in, /TYPE) ne 3 and size(file_in, /TYPE) ne 7 then begin
    print, 'usage: ULY_SPECT_READ <filename>, ...'
    print, 'first argument must be a file name or unit number'
    return, 0
endif
file_inl = file_in  # local copy of the argument
if size(file_in, /TYPE) eq 7 then begin
    if file_test(file_in) ne 1 then begin
        file_inl += '.fits'
        if file_test(file_inl) ne 1 then begin
            print, 'usage: ULY_SPECT_READ <filename>, ...'
            print, 'Error, file does not exist (' + file_in + ')'
            return, 0
        endif
    endif
endif

if n_elements(sg) eq 1 then begin
    if abs(sg) gt 10 then message, /INFO, $
      'Notice that the SG (redshift) has an odd value: '+strtrim(sg,2)+$
      ' is it correct? (it should be a "z", not a "cz")'
endif

# read the first non-empty extension that we hope contains a spectrum
naxis = 0
nhdu = 0
status = 0
while naxis eq 0 and status eq 0 do begin       # skip leading empty HDUs
    data = mrdfits(file_inl, nhdu, h, /SILENT, STATUS=status)
    if n_elements(h) eq 0 then begin
        print, 'Cannot access to the data in the file (invalid format?)'
        return, 0
    endif
    naxis = sxpar(h,'NAXIS')
    nhdu++
endwhile
if status ne 0 then begin
    print, 'Could not find a valid HDU in ', file_inl
    return, 0
endif

# switch to the appropriate reading routine
case uly_spect_filefmt(h) of
    1 : spect = uly_spect_read_sdss(data, h, ERR_SP=err_sp, SNR_SP=snr_sp, QUIET=quiet)
    2 : spect = uly_spect_read_tbl(data, h, ERR_SP=err_sp, SNR_SP=snr_sp, QUIET=quiet)
    3 : spect = uly_spect_read_lss(data, h, DISP_AXIS=disp_axis, QUIET=quiet)
    else : begin
        return, uly_spect_alloc(TITLE=file_in)
    end
endcase

ntot = n_elements(*spect.data)

if ntot eq 0 then begin
    print, 'No data read from FITS file'
    return, spect
endif

spect.title = file_in

# Handle the case when error  (or signal to noise) is read from another file
#    assume WCS are the same for error & data spectra
if n_elements(err_sp) gt 0 then begin
    testfile = FILE_INFO(err_sp)    
    if testfile.exists eq 1 then begin
        fits_read, err_sp, *spect.err, h_err 
        if disp_axis eq 2 then err = transpose(err)
    endif else $
       message, 'File:' + err_sp + ' does not exsists...'
    for i=0,n_elements((*spect.err)[0,*])-1 do begin #for 2D case
       nans = where(finite((*spect.err)[*,i]) eq 0, cnt, COMP=fin)
       if cnt ne 0 then (*spect.err)[nans,i] = max((*spect.err)[fin,i])
    endfor
endif else if keyword_set(snr_sp) then begin
    testfile = FILE_INFO(snr_sp)
    if testfile.exists eq 1 then begin
        fits_read, snr_sp, err
        if disp_axis eq 2 then err = transpose(err)
        *spect.err = *spect.data / err
    endif else $
      message, 'SNR spectrum file not valid'
    for i=0,n_elements((err)[0,*])-1 do begin #for 2D case
       neg = where(err[*,i] le 0, c, COMPLEM=pos)
       if c gt 0 then begin
          err[neg] = 1
          em = 10 * max((*spect.err)[pos,i])
          (*spect.err)[neg,i] = em
          message, 'The SNR spectrum '+strtrim(string(i),2)+ ' has '+strtrim(string(c),2)+$
                   ' negative or null values. Their error is set to '+$
                   strtrim(string(em),2), /INFO
       endif 
       nans = where(finite((*spect.err)[*,i]) eq 0, cnt, COMP=fin)
       if cnt ne 0 then (*spect.err)[nans,i] = max((*spect.err)[fin,i])
    endfor    
 endif

if n_elements(msk_sp) gt 0 then begin
    message, 'Read MASK spectrum ... NOT YET IMPLEMENTED'
endif

# Apply the shift to restframe if required
if n_elements(sg) gt 0 then begin  # shift to rest-frame
    z1 = 1d + sg
    case spect.sampling of
        0 : begin
            spect.start /= z1
            spect.step /= z1
        end
        1 : spect.start -= alog(z1)
        2 : *spect.wavelen /= z1
    endcase
endif 

# Determine wavelength range in signal spectrum and extract the required region
if n_elements(lmin) gt 0 or n_elements(lmax) gt 0 then begin
    wr = uly_spect_get(spect, /WAVERANGE)
    if n_elements(lmin) gt 0 then wr[0] = min(lmin)
    if n_elements(lmax) gt 0 then wr[1] = max(lmax)
    if n_elements(velscale) eq 1 then begin
        wr[0] *= 1D - velscale/299792.458D/2D
        wr[1] *= 1D + velscale/299792.458D/2D
    endif
    spect = uly_spect_extract(spect, WAVERANGE=wr, /OVERWRITE)
endif

ntot = n_elements(*spect.data)  

# Check and eventually replace (with 0) the NaNs, and put in goodpixels
#   (we must be sure there is no NaNs before rebinning)
good = where(finite(*spect.data), cnt, COMPLEM=nans, NCOMPLEM=nnans)
if nnans gt 0 then begin
    if not keyword_set(quiet) then $
      print, 'The input spectrum contains'+string(n_elements(nans))+' NaNs ...'
    if n_elements(*spect.goodpix) eq 0 then *spect.goodpix = good $
    else begin
       maskI = bytarr(ntot)
       maskI[*spect.goodpix] = 1
       maskI[nans] = 0
       *spect.goodpix = where(maskI eq 1)
    endelse

#   patch 1 pix of the nan regions by replicating the edge value
#   in order to reduce the oscillations of the spline interpolation
    next = nans+1
    if next[n_elements(next)-1] eq n_elements(*spect.data) then $
      next[n_elements(next)-1] = nans[n_elements(next)-1]
 #we connect here the last and the firs pixels in 2D data (not good)
    (*spect.data)[nans] = (*spect.data)[next]
    nans = where(finite(*spect.data) eq 0, cnt)
    if cnt gt 0 then begin
        prev = nans - 1
        if prev[0] lt 0 then prev[0] = nans[1]
        (*spect.data)[nans] = (*spect.data)[prev]
    endif
    nans = where(finite(*spect.data) eq 0, cnt)

    if cnt gt 0 then (*spect.data)[nans] = 0
endif

# do the same with the error, if exists
if n_elements(*spect.err) gt 0 then begin
    good = where(finite(*spect.err), cnt, COMPLEM=nans)
    if cnt eq 0 then begin
        if not keyword_set(quiet) then begin
            print, 'The error spectrum does not contain finite values'
            print, '... ignore it (ie. do as if no errors were given)'
        endif
        undefine, *spect.err
    endif else if cnt lt n_elements(*spect.err) then begin
        if not keyword_set(quiet) then $
          print, 'The input spectrum contains'+string(n_elements(nans))+' NaNs ...'
        if n_elements(*spect.goodpix) eq 0 then *spect.goodpix = good $
        else begin
           maskI = bytarr(ntot)
           maskI[*spect.goodpix] = 1
           maskI[nans] = 0
           *spect.goodpix = where(maskI eq 1, cnt)
           if cnt eq 0 then begin
               if not keyword_set(quiet) then $
                 print, 'No good pixels were left ... mask is unset'
               undefine, *spect.goodpix
           endif
        endelse
        
#   patch 1 pix of the nan regions by replicating the edge value
#   in order to reduce the oscillations of the spline interpolation
        next = nans + 1
        if next[n_elements(next)-1] eq n_elements(*spect.err) then begin
            if n_elements(nans) gt 1 then begin
                nans = nans[0:n_elements(nans)-2]
                next = next[0:n_elements(nans)-1]
            endif else next = nans
        endif 
        (*spect.err)[nans] = (*spect.err)[next]

        nans = where(finite(*spect.err) eq 0, cnt)
        if cnt gt 0 then begin
            prev = nans - 1
            if prev[0] lt 0 then begin
               nans = nans[1:*]
               prev = prev[1:*]
            endif
            (*spect.err)[nans] = (*spect.err)[prev]
        endif
        nans = where(finite(*spect.err) eq 0, cnt)
        
        if cnt gt 0 then (*spect.err)[nans] = 0
    endif
endif

# rebin in log of wavelength
if n_elements(velscale) ne 0 then begin
    if n_elements(lmin) eq 0 then $
      spect = uly_spect_logrebin(spect, velscale, /OVER) $
    else $
      spect = uly_spect_logrebin(spect, velscale, WAVERANGE=lmin[0], /OVER)
endif

#after rebining different number of pxs
ntot = n_elements(*spect.data)
dim = size(*spect.data, /DIM)

# select goodpixels the good pixels are between lmin[i] and lmax[i] 
npix = (size(*spect.data))[1]
Pix_gal = spect.start + lindgen(npix) * spect.step
#Pix_gal = range(spect.start, spect.start+(npix-1d)*spect.step,npix)

if n_elements(lmin) eq 0 then lmn = [spect.start] else begin
    lmn = lmin
    if spect.sampling eq 1 then lmn = alog(lmn) 
endelse
if n_elements(lmax) eq 0 then lmx = [spect.start+spect.step*(npix-1)] else begin
    lmx = lmax
    if spect.sampling eq 1 then lmx = alog(lmx) 
endelse

good = 0L
for i = 0, n_elements(lmn) - 1 do begin
    good = [good, $
            where((Pix_gal gt lmn[i]) and (Pix_gal lt lmx[i]))]
endfor 

if (n_elements(*spect.goodpix) eq 0) and (n_elements(good) gt 1) then begin
   maskI = bytarr(ntot)
   maskI = reform(maskI, dim)
   maskI[good[1:*], *] = 1   
   *spect.goodpix = where(maskI eq 1)
endif else begin                # have to combine with the previous mask
   maskI = bytarr(ntot)
   maskI[*spect.goodpix] = 1
   maskI = reform(maskI, dim)
   maskI[good[1:*],*] += 1
   *spect.goodpix = where(maskI eq 2)
endelse

sxaddpar, *spect.hdr, 'HISTORY', 'uly_spect_read, '''+strtrim(file_in,2)+'''

return, spect

end

#== end =======================================================================
