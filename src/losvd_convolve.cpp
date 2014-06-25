#include <Python.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <boost/python.hpp>

namespace py = boost::python;

void convolve(py::object  x, py::object y ,py::object losvd,py::object out){
  Py_Initialize();
  //convolution params
  int  i1, i2, m1, m2, KernelLen, j,HalfKernelLen ;
  int SignalLen = py::extract<int>(y.attr("__len__")());
  float *Signal,*Kernel, *param, *temp_kern;
  Signal = (float *) malloc(SignalLen * sizeof(float));
  Kernel = (float *) malloc(SignalLen * sizeof(float));
  param = (float *) malloc(4 * sizeof(float));
  temp_kern = (float *) malloc(SignalLen * sizeof(float));
  float Result[SignalLen];
  // gauss kernel params
  const float c = 299792.458;
  float vel_scale, logl_simga, norm, diff_wave;
  float *Y, *wave, *temp_con ;
  Y = (float *) malloc(SignalLen * sizeof(float));
  wave = (float *) malloc(SignalLen * sizeof(float));
  temp_con = (float *) malloc(SignalLen * sizeof(float));
  KernelLen = -99999;
  //turn python params into c arrays
  for (int i = 0 ;i<SignalLen;i++){
    Signal[i] = py::extract<float>(y.attr("__getitem__")(i));
    wave[i] = py::extract<float>(x.attr("__getitem__")(i));
    }
  for (int i = 0;i<4;i++){
    param[i] = py::extract<float>(losvd.attr("__getitem__")(i));
    }
  //start convolution
  for (int n = 0; n < SignalLen; n++)
    {  
      //calculate new kernel
      if (n == 0){
	diff_wave = (wave[n + 1] - wave[n]);
      }
      
      vel_scale = c * diff_wave / wave[n] ;
      //} else {
      //	diff_wave = (wave[n] - wave[n - 1]);
      //	vel_scale = c * diff_wave / wave[n] ;
      // } 
      logl_simga = std::log(1 + param[0]/c) / vel_scale * c;
      norm = 0;
      if (KernelLen != 2 * std::ceil(  5 * logl_simga) + 1){
	HalfKernelLen = std::ceil(  5 * logl_simga);
	KernelLen = 2 * HalfKernelLen + 1;
	//do calculation
	j = 0;
	for (int i = - HalfKernelLen  ; i <  KernelLen - HalfKernelLen; i++){
	  Y[j] = i/logl_simga;
	  Kernel[j] = std::exp(-Y[j] * Y[j]/2.);
	  if (param[2] != 0 and param[3]!= 0){
	    Kernel[j] *= (1.+ param[2] * pow(2,0.5) / pow(6,0.5) * (2 * pow(Y[j],3) - 3 * Y[j]) + param[3] / pow(24,.5) * (4 * pow(Y[j],4) - 12 * pow(Y[j],2) + 3));
	  }
	  j++;
	}
      }
	
      //do convolution
      Result[n] = 0;
      //make sure kernel matches with signal
      m2 = n + (KernelLen - 1)/2 + 1;
      m1 = n - (KernelLen - 1)/2 ;
      if (m1 < 0){
	m1 = 0;
	i1 = (KernelLen - 1)/2 - n;
	i2 = KernelLen ;
      } else if (m2 > SignalLen - 1){
	m2 = SignalLen - 1;
	i1 = 0;
	i2 = KernelLen - ((KernelLen  - 1)/2 - m2 + n) - 1;
      } else{
	i1 = 0;
	i2 = KernelLen ;
      }
      j=0;
      for (int i = i1; i <= i2; i++){
	if (Kernel[i] < 0){
	  Kernel[i] = 0;
	}
	temp_kern[j] = Kernel[i];
	norm += temp_kern[j];
	j++;
      }
      j=0;
      for (int k = m1; k <= m2; k++){
	//std:: cout << temp_kern[j] << "\t";
	temp_con[j] = Signal[k] * temp_kern[j]/norm;
	j++;
      }
      
     //do trapizode to integrate
      j = 0;
      for (int k = m1; k < m2; k++){
	Result[n] += (wave[k + 1] - wave[k]) * (temp_con[j+1] + temp_con[j])/2.;
	j++;
      }
      out.attr("__setitem__")(n,Result[n]);
    }
  //free mem
  free(Kernel);  free(Signal); free(Y); free(wave); free(temp_con); free(param);free(temp_kern);
  return ;
}

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
using namespace boost::python;
 
BOOST_PYTHON_MODULE(losvd_convolve)
{
    def("convolve", convolve);
}
