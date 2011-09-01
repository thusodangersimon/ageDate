#!/usr/bin/env python
#Thuso simon 30/8/11

import pyfits as fits
import numpy as nu
import os,sys

def comb_fits(inpath,wcs_path):
    #reads files from file and combines them based on their WCS
    #output is comb_RA_Dec.fits where RA and Dec are the center of image
    files=os.listdir(inpath)
    if inpath[-1]!='/':
        inpath=inpath+'/'
    #load fits images
    image=[]
    for i in files:
        image.append(fits.open(inpath+i))

    #create out array for image
    #1=RA 2=DEC
    RA,DEC={},{}
    for j,i in enumerate(image):
        row,col=nu.meshgrid(range(1,i[0].data.shape[1]+1),range(1,i[0].data.shape[0]+1))
        if i[0].header['CTYPE1']=='RA---TAN':
            RA[str(j)]=i[0].header['CRVAL1']+i[0].header['CD1_1']*(row  - i[0].header['CRPIX1'])+ i[0].header['CD1_2']*(col-i[0].header['CRPIX2'])
            DEC[str(j)]=i[0].header['CRVAL2']+i[0].header['CD2_1']*(row  - i[0].header['CRPIX1'])+ i[0].header['CD2_2']*(col-i[0].header['CRPIX2'])
        else:
            DEC[str(j)]=i[0].header['CRVAL1']+i[0].header['CD1_1']*(row  - i[0].header['CRPIX1'])+ i[0].header['CD1_2']*(col-i[0].header['CRPIX2'])
            RA[str(j)]=i[0].header['CRVAL2']+i[0].header['CD2_1']*(row  - i[0].header['CRPIX1'])+ i[0].header['CD2_2']*(col-i[0].header['CRPIX2'])
           




    #create image with all fields
        '''
row=i[0].header['CRPIX1']+ (i[0].header['CD2_1']*i[0].header['CD1_2'] -i[0].header['CD1_1']*i[0].header['CD2_2'])/(i[0].header['CD1_1']*i[0].header['CRVAL2'] + i[0].header['CD2_1']*RA - i[0].header['CD1_1']*DEC - i[0].header['CD2_1']*i[0].header['CRVAL1'])
col= i[0].header['CRPIX2']+(i[0].header['CD1_2']*i[0].header['CRVAL2'] + i[0].header['CD2_2']*RA - i[0].header['CD1_2']*DEC - i[0].header['CD2_2']*i[0].header['CRVAL1'])/(i[0].header['CD1_1']*i[0].header['CRVAL2'] + i[0].header['CD2_1']*RA - i[0].header['CD1_1']*DEC - i[0].header['CD2_1']*i[0].header['CRVAL1'])

'''
