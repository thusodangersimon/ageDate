from mpi4py import MPI as mpi
import cPickle as pik
import pandas as pd
import pyfits as fits
import numpy as np
import emcee_lik, emcee
import zmq
import os, sys
import database_utils as util
import time
import socket

'''takes data from LRG.pik and processes the fits files into data for MCMC
fitting'''

def get_data(db_path):
    '''retrives data from db.
    returns matrix of params (ssp, n param), spec (ssp, wavelength)'''

    db = util.numpy_sql(db_path)
    # get table name
    table_name = db.execute('select * from sqlite_master').fetchall()[0][1]
    # fetch all
    spec, param = [] ,[]
    for imf, model, tau, age, metal, buf_spec in db.execute('SELECT * From %s'%table_name):
        spec.append(util.convert_array(buf_spec)[:,1])
        param.append([tau, age, metal])
    
    param = np.array(param)
    spec = np.array(spec)
    return param, spec, util.convert_array(buf_spec)[:,0]

def fits2data(fitsfile, sn):
    '''Returns nx3 data array from fits spectra'''
    if isinstance(fitsfile, str):
        # turn into buffer and load
        fitsfile = fits.open(io.BytesIO(fitsfile))
    step = fitsfile[0].data.shape[-1] * fitsfile[0].header['CD1_1']
    maxwave = fitsfile[0].header['CRVAL1'] + step
    wavelength = np.linspace(fitsfile[0].header['CRVAL1'], maxwave,
                             fitsfile[0].data.shape[-1])
    data = np.zeros((len(wavelength), 3))
    data[:,0] = wavelength
    data[:,1] = fitsfile[0].data[0,:][0]
    data[:,2] = np.median(fitsfile[0].data[0,:][0])/ sn
    return data

class Server(object):
    def __init__(self, send_path, result_path, sendport=5557, reciveprot=5558,
                 mangeport=5556):
        '''Initalizes server on specific ports to act like flags for telling
           processor what to do.
           
           :param send_path: path to look for galaxies to send

           :param result_path: path to put results once finished. Also will check here when restarting after a crash.

           :param sendport: port to use for sending data to worker

           :param reciveprot: port to recive data from workers

           :param mangeport: port used to track progress of fitting with each different instance.'''
        self.send_path = send_path
        self.result_path = result_path
        self.sendport = int(sendport)
        self.recivport = int(reciveprot)
        self.manageport = int(mangeport)
        self.context = zmq.Context()
        self.cur_workers = {}
        self.initalize()
        # other varibles
        self.done = False
        if os.path.exists('complete_log.pik'):
            self.complete = pik.load(open('complete_log.pik'))
        else:
            self.complete = {}
        
    def initalize(self):
        '''starts all sockets'''
        # send
        self.gal_send = self.context.socket(zmq.PUSH)
        self.gal_send.bind("tcp://*:%i"%self.sendport)
        self.gal_send.setsockopt(zmq.IDENTITY, b'gal send')
        # recive results
        self.results_receiver = self.context.socket(zmq.PULL)
        self.results_receiver.bind("tcp://*:%i"%self.recivport)
        self.results_receiver.setsockopt(zmq.IDENTITY, b'gal recive')
        # manage
        self.control_sender = self.context.socket(zmq.PUB)
        self.control_sender.bind("tcp://*:%i"%self.manageport)
        self.control_sender.setsockopt(zmq.IDENTITY, b'gal manage')
        # make poll
        self.poller = zmq.Poller()
        self.poller.register(self.results_receiver, zmq.POLLIN)
        #
        

        # load in galaxies

    def start(self):
        '''Starts distributing data from input directories'''
        while True:
            time.sleep(1)
            # check results
            socks = dict(self.poller.poll(1000))           
            if self.results_receiver in socks:
                # get results
                msg = None
                msg, id, py_obj = self.results_receiver.recv_pyobj(zmq.NOBLOCK)
                if msg == 'need data' and not self.done:
                    # Send data and record worker
                    print 'sending data to %s'%id
                    data = self.get_next()
                    self.update(id, 0)
                    self.gal_send.send_pyobj((id, data))
                elif msg == 'status':
                    # log status
                    print 'update from %s'%id
                    self.update(id, py_obj)
                elif msg == 'done':
                    # get results and send more data or tell processes to finish
                     print '%s finished'%id
                     self.finalize(id, py_obj)
            # check if finished
            if self.done:
                self.close()
                
    def check_current(self, spec_index=None):
        '''checks if index is being used or has been completed all ready.
        If spec_index is None then starts from lowest index'''
        if spec_index is None:
            if len(self.complete) == 0:
                return 0
            else:
                spec_index = 0
        # get list of working indexes
        working_index = [self.cur_workers[i][2] for i in self.cur_workers]
        while spec_index in self.complete or spec_index in working_index:
            spec_index += 1
        return spec_index
 
    def add(self, id, param, spec_index):
        '''Adds id to busy list'''
        # param, ess
        self.cur_workers[id] = [param, 0, spec_index]
        
    def get_next(self):
        '''Gets next data to send and checks how many gal are left'''
        print 'this many are left'
        self.done = True
        return None, None
    
    def update(self, id, ess):
        '''Keeps track of all data that is being processed and prints message
        about status'''
        # record status
        if id in self.cur_workers:
            self.cur_workers[id][1] = ess
        else:
            # echo from finished param?
            pass
        for ID in self.cur_workers:
            if  self.cur_workers[ID][1] > 0:
                print '%s is has ESS of %f'%(ID, int(self.cur_workers[ID][1]))
            else:
                 print '%s is in burnin'%ID
            
    def finalize(self, id):
        '''When data is finished. Saves and removes from working list'''
        print 'worker %s is done' %id
        self.complete[self.cur_workers[id][2]] = self.cur_workers[id][0]
        # save list
        pik.dump(self.complete, open('complete_log.pik' ,'w'), 2)
        self.cur_workers.pop(id)
        
    def close(self):
        '''Send mesgage for all processers to close'''
        self.control_sender.send('exit')
        self.control_sender.send('exit')

class Client(object):
    '''
# The "worker" functions listen on a zeromq PULL connection for "work" 
# (numbers to be processed) from the ventilator, square those numbers,
# and send the results down another zeromq PUSH connection to the 
# results manager.
    '''
    def __init__(self, host_addr, reciveport=5557, sendport=5558,
                 mangeport=5556):
        # get identity
        self.id = socket.gethostname() + str(int(round(np.random.rand()*1000)))
        self.sendport = int(sendport)
        self.recivport = int(reciveport)
        self.manageport = int(mangeport)
        self.host_addr = host_addr
        self.context = zmq.Context()
        # initalize contex managers
        # gets data to work on
        self.work_receiver = self.context.socket(zmq.PULL)
        self.work_receiver.connect("tcp://%s:%i"%(self.host_addr, self.recivport))
        self.work_receiver.setsockopt(zmq.IDENTITY, self.id )
        # send results and status
        self.results_sender = self.context.socket(zmq.PUSH)
        self.results_sender.connect("tcp://%s:%i"%(self.host_addr, self.sendport))
        self.results_sender.setsockopt(zmq.IDENTITY, self.id )
        # gets info when to quit
        self.control_receiver = self.context.socket(zmq.SUB)
        self.control_receiver.connect("tcp://%s:%i"%(self.host_addr,self.manageport))
        self.control_receiver.setsockopt(zmq.SUBSCRIBE, '')
        # make poller
        self.poller = zmq.Poller()
        self.poller.register(self.work_receiver, zmq.POLLIN)
        self.poller.register(self.control_receiver, zmq.POLLIN)
        
    def get_data(self):
        '''Reqests data for processing'''
        # tell client that it's ready
        self.results_sender.send_pyobj((b'need data', self.id, []))
        sock = dict(self.poller.poll(100))
        while True:
            # check is all work is done
            done = self.check_done()
            if done:
                return None, None
            try:
                id, pyobj= self.work_receiver.recv_pyobj(zmq.NOBLOCK)
                if id == self.id:
                    data, param = pyobj
                    self.results_sender.send_pyobj((b'got it', self.id, []))
                    break
            except zmq.Again:
                print 'trouble reciving data try again in 5 seconds'
                self.results_sender.send_pyobj((b'waiting', self.id, []))
                time.sleep(5)
            
        print 'recived_data'
        return data, param
    
    def send_update(self, ess):
        '''Sends effective sample size to client'''
        # send client report
        self.results_sender.send_pyobj(('status',self.id, ess))

    def send_results(self, results):
        '''When done sends results'''
        # send results
        self.results_sender.send_pyobj(('done', self.id, results))
        # check if need to quit

    def check_done(self):
        '''Checks if I should finish'''    
        socket = self.poller.poll(1000)
        if self.control_receiver in socket:
            msg = self.control_receiver.recv(zmq.NOBLOCK)
            if msg == 'exit':
                print 'Finishing'
                return True
        return False


def worker_fit(db_path, host_addr):
    '''Worker side of fitting, uses emcee for the fitting and goes until
    server tell us to stop'''
    # initalize classes
    client = Client(host_addr)
    comm = mpi.COMM_WORLD
    size = comm.size
    rank = comm.rank
    pool = emcee_lik.MPIPool_stay_alive(loadbalance=True)
    # fit till told to quit
    while True:
        if rank == 0:
            data, data_param = client.get_data()
        else:
            data, data_param = None, None
        data, data_param = comm.bcast((data, data_param), 0)
        # kill signal
        if data is None:
            sys.exit(0)
        # get postieror function
        posterior = emcee_lik.LRG_emcee({'test':data}, db_path, have_dust=True, have_losvd=False)
        posterior.init()
        nwalkers = 2 *  posterior.ndim()
        if not pool.is_master():
                # Wait for instructions from the master process.
                pool.wait(posterior)
                continue
        
        sampler = emcee.EnsembleSampler(nwalkers, posterior.ndim() ,
                                            posterior, pool=pool)
        pos0 = posterior.inital_pos(nwalkers)
        i = 0
        accept = 0
        for pos, prob, rstate in sampler.sample(pos0, iterations=500):
            show = 'Burn-in: Postieror=%e acceptance=%2.1f'%(np.mean(prob),accept)
            print show
            accept = np.mean(sampler.acceptance_fraction)
            i += 1
            if i % 100 == 0:
                client.send_update(-1)
        # get ready for real run
        sampler.reset()
        ess, accept = 0., 0.
        trys = 0
        while True:
            for pos, prob, rstate in sampler.sample(pos, iterations=100, rstate0=rstate):
                show = 'Real run: Postieror=%e acceptance=%2.1f ESS=%2.1f'%(np.mean(prob), 100*accept, ess)
                print show
                accept = np.mean(sampler.acceptance_fraction)
            ess = (sampler.flatchain.shape[0]/
                    np.nanmin(sampler.get_autocorr_time()))
            # send update
            client.send_update(ess)
            if ess >= 1000:
                # Try to keep from randomly getting an ess 
                trys +=1
                if trys > 2:
                    break
            else:
                trys = 0
        # tell workers to stop
        pool.close()
        # send results
        client.send_results((data, data_param, sampler.flatchain,
                             sampler.flatlnprobability))

def fit_all(db_path, save_path, min_wave=3500, max_wave=8000):
    '''Starts job and orgainizes data'''
    # check if save_path exsists
    assert os.path.exists(save_path), 'Please put in a valid save path'
    param, spec, wave = get_data(db_path)
    # get min and max wavelength
    index = np.where(np.logical_and(wave >= min_wave, wave <= max_wave))[0]
    wave = wave[index]
    spec = spec[:,index]
    server = Server('','')
    print 'Ready for requests'
    spec_index = server.check_current()
    while True:
        # skip index till new one is found
        if spec_index >= spec[param[:,1]>9].shape[0]:
            # don't take any more data
            server.done = True
            if len(server.cur_workers) == 0:
                print 'exiting'
                break
            server.close()
        # check results
        socks = dict(server.poller.poll(1000))
        if server.results_receiver in socks:
            # get results
            msg = None
            msg, id, py_obj = server.results_receiver.recv_pyobj(zmq.NOBLOCK)
            print msg, id
            if msg == 'need data' and not server.done:
                # Send data and record worker
                print 'sending data to %s'%id
                data = np.vstack((wave, spec[param[:,1]>9][spec_index,:])).T
                data_param = param[param[:,1]>9][spec_index]
                server.add(id, data_param, spec_index)
                spec_index += 1
                spec_index = server.check_current(spec_index)
                server.update(id, 0)
                server.gal_send.send_pyobj((id, (data, data_param)))
                
            elif msg == 'waiting':
                # if process is waiting for data resend
                try:
                    temp_index = server.cur_workers[id][2]
                    data = np.vstack((wave, spec[param[:,1]>9][temp_index,:])).T
                    data_param = param[param[:,1]>9][temp_index]
                    print 'sending again to %s'%id
                    server.gal_send.send_pyobj((id, (data, data_param)))
                except KeyError:
                    # if getting echo after has finnished
                    continue
                       
            elif msg == 'status':
                # log status
                server.update(id, py_obj)
                
            elif msg == 'done':
                # get results and send more data or tell processes to finish
                server.finalize(id)
                # save
                data_param = py_obj[1]
                out_name = '%2.2f_%2.2f_%2.2f.pik'%(data_param[0], data_param[1],
                                                data_param[2])
                pik.dump(py_obj, open(os.path.join(save_path,out_name),'w'), 2)
                spec_index = spec[param[:,1]>9].shape[0]
                continue
            
if __name__ == '__main__':
    # start worker'
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
    worker_fit(db_path, '54-4-a6-a-f4-28.lan.uct.ac.za')
