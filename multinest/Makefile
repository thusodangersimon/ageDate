PROGS = libnest3.a 
 

 
AR = ar r  
LINKLIB = ld -shared  
LIBDIR = /media/THUSO/Phd/Code/multinest
 
NSOBJECTS = utils.o utils1.o priors.o kmeans_clstr.o xmeans_clstr.o posterior.o nested.o

%.o: %.f90
	$(FC) $(FFLAGS) -c -o $@ $^ 

%.o: %.F90
	$(FC) $(FFLAGS) -c -o $@ $^ 

 
all: $(PROGS) 
#	-rm nested.o
 
libnest3.so: $(NSOBJECTS) 
	$(LINKLIB) -o $@ $^ 
#	-mv $@ $(LIBDIR)  
 
libnest3.a: $(NSOBJECTS) 
	$(AR) $@ $^ 
 
 
clean: 
	-rm -f $(LIBDIR)/libnest3.*  *.o *.mod

