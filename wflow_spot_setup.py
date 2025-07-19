import spotpy
from spotpy.parameter import *
from spotpy.objectivefunctions import rmse,nashsutcliffe,kge
import HydroErr as he
# from spotpy.examples.hymod_python.hymod import hymod
import os

from RunWflow_Julia import *
from Synthetic_obs import *

#%%
class spot_setup(object):
    #These are spotpy parameter objects so spotpy will recognize them automatically
    
    # def __init__(self, basin,
    #              run_name, 
    #              start_year, 
    #              end_year, 
    #              test =False ,
    #              obj_func = None,
    #              rootdir = "/home/pwiersma/scratch/Data/ewatercycle",
    #              parsetdir = "/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV",
    #              config_file ="sbm_config_CH_orig.toml" ,
    #              all_params_file = None,
    #              resolution = '1000m',
    #              static_pardic = None,
    #              months = 'all',
    #              spinupparsetname = None,
    #              posterior_parset_file = None, 
    #              cluster_parset = None, 
    #              skip_first_year =None,
    #              calibration_purpose ='Soilcalib',
    #              synthetic_obs =None,
    #              routing_margin =False,
    #              yearly_params = None, 
    #              prior_ranges = None,
    #              peak_filter = False,
    #              double_months = None,
    #              pbias_threshold = None,
    #              obs_error_scale = None,
    #              obsdic = None,
    #                               station_names= None
    #                             ):
    def __init__(self, 
                 config,
                 RUN_NAME,
                 calibration_purpose,
                 soil_parset = None,
                 single_year = None):
        self.cfg = config
        self.RUN_NAME = RUN_NAME
        self.calibration_purpose = calibration_purpose
        for key, value in config.items():
            setattr(self, key, value)
        for key,value in config['OBS_ERROR'].items():
            setattr(self,key,value)
        if not single_year == None:
            self.START_YEAR = single_year-1
            self.END_YEAR = single_year
        else:
            self.START_YEAR = config['START_YEAR']-1
            self.END_YEAR = config['END_YEAR']

        self.soil_parset = soil_parset
        self.OUTDIR = join(self.ROOTDIR,"outputs",f"{self.EXP_ID}")


        self.params = []
        if calibration_purpose =='Soilcalib':
            for par,value in self.param_ranges.items():
                if self.param_soilcalib[par]:
                    self.params.append(spotpy.parameter.Uniform(par, value[0],value[1],(value[0]+value[1])/2))
        
        elif calibration_purpose == 'Yearlycalib':
            for par,value in self.param_ranges.items():
                if self.param_kinds[par] in ['snow','meteo']:
                    self.params.append(spotpy.parameter.Uniform(par, value[0],value[1],(value[0]+value[1])/2))

        elif calibration_purpose == 'OB_calib':
            self.params.append(spotpy.parameter.Uniform('rfcf',self.param_ranges['rfcf'][0],
                                                        self.param_ranges['rfcf'][1],
                                                        (self.param_ranges['rfcf'][0]+self.param_ranges['rfcf'][1])/2))
        print('self.params:',self.params)

        self.wflow = RunWflow_Julia( ROOTDIR = self.ROOTDIR,
            PARSETDIR =self.EXPDIR,
            BASIN = self.BASIN,
            RUN_NAME = self.RUN_NAME,
            START_YEAR = self.START_YEAR,   
            END_YEAR = self.END_YEAR,
            CONFIG_FILE = "sbm_config_CH_orig.toml",
            RESOLUTION = self.RESOLUTION,
            CONTAINER = self.CONTAINER,
            NC_STATES = self.NC_STATES)
       
        forcing_name = f"wflow_MeteoSwiss_{self.wflow.RESOLUTION}_{self.BASIN}_{self.START_YEAR-1}_{self.END_YEAR}.nc"
        staticmaps_name = f"staticmaps_{self.wflow.RESOLUTION}_{self.BASIN}_feb2024.nc"

        self.wflow.generate_MS_forcing_if_needed(forcing_name)
        self.wflow.check_staticmaps(staticmaps_name)
        self.SKIPFIRSTYEAR =1
        self.wflow.load_stations([self.BASIN])

    def parameters(self):
        #smaple from spotpy parameter objects
        newparams = spotpy.parameter.generate(self.params)
        return newparams
    
    def simulation(self,x):
        #Load smapled parameter sets 
        #for soilcalib: add fixed parameters
        #for yearly calib: 

        #x is a spotpy.parameter.ParameterSet object
        #You acces the name through x.name, and the value through x['parname'] or just x[0]
        #Use spotpy.parameters package 
        # pars = spotpy.parameter.get_parameters_array(self) #<-- don't do this
        pardic = {}

        #figure out mpi rank 
        try:
            from mpi4py import MPI
            self.rank = MPI.COMM_WORLD.Get_rank()
            size = MPI.COMM_WORLD.Get_size()
            print("MPI size = ", size)
            print("MPI rank = ", self.rank)
        except:
            self.rank = 0

        sample_name = f"sample_{self.rank}_{self.RUN_NAME}"
        pardic[sample_name] = self.param_fixed.copy()
        if self.calibration_purpose == 'OB_calib':
            pardic[sample_name][self.OSHD] = True
            pardic[sample_name]['DD'] = 'static'
            pardic[sample_name]['sfcf'] = 0
            pardic[sample_name]['masswasting'] = False
            


            
        if not self.soil_parset == None:
            for par,value in self.soil_parset.items():
                pardic[sample_name][par] = value
            # print(pardic)

        #for yearly calib, we need to load the k_soil clusters coming from the soilcalib and save them
        # as fixed parameters

        #TODO remove this stuff from spotpy
        if np.all(x) == 0:
            print("The first run correctly takes the original parameters")
            parnames = x.name
            randompars = spotpy.parameter.generate(self.params)['random']
            for i,par in enumerate(parnames):
                # print(par,x[i])
                pardic[sample_name][par] = randompars[i]  
        else:
            for i,par in enumerate(x.name):
                pardic[sample_name][par] = x[i]
        
        print(pardic)

        print("About to adjust config in wflow_julia_spotpy")#, pardic[sample_name])
        self.wflow.load_model_dics(pardic)
        self.wflow.adjust_config()
        initialize_time = time.time()
        self.wflow.create_models()
        print(f"Initialization takes {time.time() - initialize_time } seconds")

        self.wflow.series_runs(self.wflow.standard_run,test = self.TEST)


        self.wflow.finalize_runs()
        self.wflow.remove_forcing()


        self.wflow.load_Q()
        self.wflow.stations_combine()

        if self.SAVE_SPOTPY_SWE ==True and self.calibration_purpose=='Yearlycalib': 
            SWEdic = self.wflow.load_SWE()
            SWE = SWEdic[sample_name]
            SWE = SWE.sel(time = slice(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",
                                                            f'{self.wflow.END_YEAR}-09-30'))
            if self.CONTAINER in [None,'' ]:
                sm = join(self.ROOTDIR,"experiments/data/input",f"staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc")
            else:
                sm = join(self.CONTAINER,f"staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc")
            dem = xr.open_dataset(sm)['wflow_dem'].sortby('lat')
            SWE = xr.where(~np.isnan(dem),SWE,np.nan)

            # member= self.RUN_NAME.split('_')[2][2]
            splitted = self.RUN_NAME.split('_')
            for splitte in splitted:
                if splitte.startswith('ms'):
                    member = splitte[2:]
            SWEdir = join(self.OUTDIR, f"Yearlycalib", f"{self.END_YEAR}", member,"spotpy_SWE")

            Path(SWEdir).mkdir(parents = True, exist_ok = True)
            identifier = str(time.time())
            for key,value in pardic[sample_name].items():
                if isinstance(value, dict):
                    continue 
                SWE.attrs[key] = str(value)
            SWE.to_netcdf(join(SWEdir,f"SWE_{identifier}.nc"))

            if self.SAVE_NC_STATES_SPOTPY ==True:
                outdir = self.wflow.MODEL_DICS[sample_name]['config']['output']['path']
                outfile = join(self.ROOTDIR, "experiments",
                    self.wflow.MODEL_DICS[sample_name]['config']['dir_output'],outdir)
                NC_STATES_dir = join(self.OUTDIR, f"Yearlycalib", f"{self.END_YEAR}", member,"spotpy_NC_STATES")
                Path(NC_STATES_dir).mkdir(parents = True, exist_ok = True)
                with xr.open_dataset(outfile) as ncfile:
                    for state in self.NC_STATES:
                        if state in ['snow','snowwater']:
                            continue
                        print(state, NC_STATES_dir)
                        state_array = ncfile[state]
                        state_array.to_netcdf(join(NC_STATES_dir,f"{state}_{identifier}.nc"))


        if self.SWE_CALIB ==False:
            global output_sim 
            out_list = []
            for name in [self.BASIN]:
                out_df =self.wflow.stations[name].combined[sample_name][
                    slice(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",f'{self.wflow.END_YEAR}-09-30')
                    ]
                if self.TEST:
                    out_df.loc[:]= np.random.normal(10,0.5,out_df.shape).astype(np.float32)
                # if not self.SNOWMONTHS ==None:
                #     mask = out_df.index.month.isin(self.SNOWMONTHS)
                #     out_df = out_df[mask]
                out_item = out_df.values
                # self.nans = np.isnan(out_item)
                out_item = out_item[~np.isnan(out_item)] #mask where obs have nans? 
                
                out_list.append(out_item)
            output_sim = np.concatenate(out_list)
                
            print('sim',len(output_sim))
            print('Simulation finished')
            return output_sim

        elif self.SWE_CALIB ==True:
            SWEdic = self.wflow.load_SWE()
            SWE = SWEdic[sample_name]
            SWE = SWE.sel(time = slice(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",
                                                            f'{self.wflow.END_YEAR}-09-30'))
            if self.CONTAINER in [None,'' ]:
                sm = join(self.ROOTDIR,"experiments/Data/input",f"staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc")
            else:
                sm = join(self.CONTAINER,f"staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc")            
            dem = xr.open_dataset(sm)['wflow_dem'].sortby('lat')
            SWE = xr.where(~np.isnan(dem),SWE,np.nan)

            SWEdif = SWE.diff('time')
            if SWEdif.shape != self.sweshape:
                print("Shape is not the same")
                print(SWEdif.shape)
                print(self.sweshape)

            swe_array = SWEdif.data.flatten()
            # swe = swe_array[~np.isnan(swe_array)]
            output_sim = swe_array
            if self.TEST == True:
                daterange = pd.date_range(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",
                                                    f'{self.wflow.END_YEAR}-09-30')
                shape = [SWE.lat.shape[0],SWE.lon.shape[0]]
                output_sim = np.random.normal(10,0.5,
                                              size = (len(daterange),shape[0],shape[1])).astype(np.float32)
                output_sim = np.where(np.isnan(dem),np.nan,output_sim)
                output_sim = np.diff(output_sim,axis = 0)
                output_sim = output_sim.flatten()
                # output_sim = output_sim[~np.isnan(output_sim)]
            print('sim',len(output_sim))
            print('Simulation finished')
            return output_sim

    
    def evaluation(self):
        if self.SETTING =='Real':
            print('Gathering observations in evaluation()')
            global output_eval
            # out_list = []
            # for name in self.station_names:
                
            out_df= self.wflow.stations[self.BASIN].obs[slice(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",
                                                        f'{self.wflow.END_YEAR}-09-30')]
            # if not self.SNOWMONTHS ==None:
            #     mask = out_df.index.month.isin(self.SNOWMONTHS)
            #     out_df = out_df[mask]
            # out_list.append(out_df.values.flatten())
            output_eval = out_df.values.flatten() #np.concatenate(out_list)
        
            print('obs',output_eval.shape)
        elif self.SETTING =='Synthetic':
            #TODO make it work also for other upstream stations
            print('Gathering synthetic observations in evaluation()')
            # synthetic_dir = join(self.wflow.ROOTDIR,"experiments/wflow_julia_1000m_2005_2015_JLLRV/synthetic_obs",
            #                      f"{self.synthetic_obs}")
            # synthetic_dir = join(self.OUTDIR,"Synthetic_obs")
            synthetic_dir = self.SYNDIR

            if self.SWE_CALIB ==False:
                if self.OBS_ERROR!= None:
                    if self.OBS_ERROR['repeat'] == False:
                        print("using single obs noise")
                        out_df = pd.read_csv(join(synthetic_dir,f"Q_{self.EXP_NAME}_{self.OBS_ERROR['scale']}.csv"),
                                            index_col = 0,parse_dates=True)
                #TODO make it work for other synthetic cases 
                # if self.obs_error_scale!=None:
                #     print("using obs noise scale")
                #     out_df = pd.read_csv(join(synthetic_dir,f"Q_{run_name}_{self.obs_error_scale}.csv"),index_col = 0)
                #     # out_df = pd.read_csv(join(synthetic_dir,f"Q_synthetic_{self.BASIN}_{self.synthetic_obs}_obsscale_{self.obs_error_scale}.csv"),index_col = 0)
                # else:
                #     out_df = pd.read_csv(join(synthetic_dir,f"Q_synthetic_{self.BASIN}_{self.synthetic_obs}.csv"),index_col = 0)
                    
                out_df = out_df[slice(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",
                                                            f'{self.wflow.END_YEAR}-09-30')]
                # if not self.SNOWMONTHS ==None:
                #     mask = out_df.index.month.isin(self.SNOWMONTHS)
                #     out_df = out_df[mask]
                output_eval = out_df.values.flatten()
                print('synthetic obs',output_eval.shape)


            elif self.SWE_CALIB ==True:
                if self.SYN_SNOWMODEL == 'OSHD':
                    C = SnowClass(self.BASIN)
                    if self.OSHD == 'EB':
                        SWE = C.load_FSM(resolution = self.RESOLUTION, root = self.CONTAINER)['swet_all']
                    elif self.OSHD =='TI':
                        SWE = C.load_OSHD(resolution = self.RESOLUTION,root = self.CONTAINER)['swee_all']
                elif self.SYN_SNOWMODEL == 'wflow':
                    SWE = xr.open_dataset(join(synthetic_dir,f"SWE_{self.EXP_NAME}.nc"))
                SWE = SWE.sel(time = slice(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",
                                                            f'{self.wflow.END_YEAR}-09-30'))
                SWE = SWE.rename({'x':'lon','y':'lat'})
                if self.CONTAINER in [None,'' ]:
                    sm = join(self.ROOTDIR,"experiments/Data/input",f"staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc")
                else:
                    sm = join(self.CONTAINER,f"staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc")
                dem = xr.open_dataset(sm)['wflow_dem']                
                SWE = xr.where(~np.isnan(dem),SWE,np.nan)
                
                SWEdif = SWE.diff('time')
                self.sweshape = SWEdif.shape

                # swe_df = pd.read_csv(join(synthetic_dir,f"SWE_{self.EXP_NAME}.csv"),index_col = 0,parse_dates=True)
                # SWE = 
                # swe_df = swe_df[slice(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",
                #                                             f'{self.wflow.END_YEAR}-09-30')]
                swe_array = SWEdif.data.flatten()
                # swe_array = swe_array[~np.isnan(swe_array)]
                output_eval = swe_array
                print('synthetic obs',output_eval.shape)
                
        return output_eval  #1461
         
    def objectivefunction(self,simulation,evaluation,params = None, return_pbias = False,plot=False,
                          return_filtered = False):
        
        if self.OF =='LOA':
            signatures = self.LOA_signatures
            

        if self.SWE_CALIB ==True:
            # rmse = he.rmse(evaluation,simulation)

            # sm = join(self.ROOTDIR,"experiments/Data/input",f"staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc")
            # dem = xr.open_dataset(sm)['wflow_dem']  
            # dim1,dim2 = dem.shape
            # dim3 = len(evaluation)//(dim1*dim2)
            # simulation = np.reshape(simulation,(dim3,dim1,dim2))
            # evaluation = np.reshape(evaluation,(dim3,dim1,dim2))
            # dsim = np.diff(simulation,axis = 0)
            # deval = np.diff(evaluation,axis = 0)

            # OF  = kge(dsim.flatten(),deval.flatten())

            #mask where simulationa or evaluation has a nan
            print("simshape",simulation.shape)
            print("evalshape",evaluation.shape)
            mask = np.isnan(evaluation) | np.isnan(simulation)
            simulation = simulation[~mask]
            evaluation = evaluation[~mask]
            print("simshape",simulation.shape)
            print("evalshape",evaluation.shape)
            posdif_sim = (np.diff(simulation,prepend=0)>0) * np.diff(simulation,prepend=0)
            posdif_eval = (np.diff(evaluation,prepend=0)>0) * np.diff(evaluation,prepend=0)
            print("simsum",np.sum(simulation))
            print("evalsum",np.sum(evaluation))
            OF = -he.rmse(posdif_eval,posdif_sim)

            pbias = (np.sum(simulation) - np.sum(evaluation))/np.sum(evaluation)*100 
            if return_pbias and return_filtered:
                return OF, [simulation,evaluation], pbias 
            elif return_pbias:
                return OF, pbias
            elif return_filtered:
                return OF, [simulation,evaluation]
            else:
                return OF
        else:
            daterange = pd.date_range(f"{self.wflow.START_YEAR-1 +self.SKIPFIRSTYEAR}-10-01",
                                                    f'{self.wflow.END_YEAR}-09-30')
            
            if (self.calibration_purpose=='Soilcalib') and (self.SNOWFREEMONTHS != None):
                print("Removing snow months")
                mask = np.array([date.month in self.SNOWFREEMONTHS for date in daterange])
                simulation = np.array(simulation)[mask].tolist()
                evaluation = np.array(evaluation)[mask].tolist()
                daterange = daterange[mask]
                print("Snowfree months",len(evaluation),len(simulation))
            elif (self.calibration_purpose=='Yearlycalib') and (self.SNOWMONTHS != None):
                print("Keeping only snow months")
                mask = np.array([date.month in self.SNOWMONTHS for date in daterange])
                simulation = np.array(simulation)[mask].tolist()
                evaluation = np.array(evaluation)[mask].tolist()
                daterange = daterange[mask]
                print("Snow months",len(evaluation),len(simulation))
        
            if self.OBS_ERROR['repeat'] == True:
                print("using repeated obs noise")
                print("Clean sum", np.sum(evaluation))
                if self.OBS_ERROR['kind'] == 'additive':
                    evaluation = additive_gaussian_noise(evaluation,self.OBS_ERROR['scale'])
                elif self.OBS_ERROR['kind'] == 'multiplicative':
                    evaluation = multiplicative_gaussian_noise(evaluation,self.OBS_ERROR['scale'])
                print("Contaminated sum", np.sum(evaluation))
            if (self.calibration_purpose == 'Yearlycalib') and (self.PEAK_FILTER ==True): 
                # print("Peak filtering")
                def peakfilter(array):
                    diff = np.diff(array,prepend=0)
                    posdif = (diff>0) * diff
                    mask1 = posdif> 2*np.std(posdif)
                    mask1_shifted = np.roll(mask1, -1)
                    mask1_shifted2 = np.roll(mask1, -2)
                    mask1_shifted3 = np.roll(mask1, -3)
                    # Create a new mask that includes the day after each storm peak
                    mask2 = mask1 | mask1_shifted | mask1_shifted2 | mask1_shifted3
                    mask3 = array > np.quantile(array, 0.95)
                    mask = mask2 | mask3
                    return mask
                obs = np.array(evaluation)
                sim = np.array(simulation)
                
                obsmask = peakfilter(obs)
                simmask = peakfilter(sim)
                mask = obsmask | simmask
                #undo mask for self.doublet months
                # months = self.SNOWMONTHS
                
                if not self.DOUBLE_MONTHS == None:
                    monthmask = ~np.array([date.month in self.DOUBLE_MONTHS for date in daterange])
                    # print("Monthmask",len(monthmask))
                    # print("Mask",len(mask))
                    mask = mask & monthmask


                #peakfilterplot = f"/home/pwiersma/scratch/Figures/ewc_figures/{self.BASIN}_peakfilterr.png"

                # if not os.path.exists(peakfilterplot):
                #     rangea = slice(0,3000)
                #     obs_df = pd.Series(obs,index = daterange)
                #     unfiltered_obs = obs_df[rangea]
                #     filtered_obs = obs_df[~mask][rangea]
                    
                    
                #     f1,ax1 = plt.subplots(figsize = (12,7))
                #     unfiltered_obs.plot(ax = ax1,label = 'Unfiltered')
                #     filtered_obs.plot(ax = ax1, label = 'Filtered')
                #     ax1.set_ylabel('Q [m3/s]')
                #     ax1.grid()
                #     plt.legend()
                #     ax1.set_title(f"{self.BASIN} rainfall peak filter")
                #     f1.savefig(peakfilterplot,
                #                 dpi = 250, bbox_inches = 'tight')

                
                evaluation = obs[~mask].tolist()
                simulation = sim[~mask].tolist()    

                single_eval = np.array(evaluation)
                single_sim = np.array(simulation)

                daterange = daterange[~mask]
        
            if (self.PBIAS_THRESHOLD != None) or (return_pbias == True):        
                # print("Calculating pbias")   
                pbiases = []
                for year in range(self.wflow.START_YEAR,self.wflow.END_YEAR+1):
                    # yearmask = (daterange.year == year).tolist()
                    year_data_obs = np.array(evaluation)[daterange.year == year]#[yearmask==True]
                    year_data_sim = np.array(simulation)[daterange.year == year]
                    year_pbias = ((year_data_sim.sum() - year_data_obs.sum())/year_data_obs.sum())*100
                    pbiases.append(year_pbias)
                    #TODO take the max of the absolute pbias obviously 
                pbias = np.max(pbiases)
            if not self.DOUBLE_MONTHS ==None:
                # print("Double months first ",len(evaluation),len(simulation))    
                dmonths = self.DOUBLE_MONTHS
                # duplicates = self.duplicat

                obs_df = pd.DataFrame(evaluation,index = daterange)
                sim_df = pd.DataFrame(simulation,index = daterange)
                final_obs_list  = []
                final_sim_list  = []

                for year in obs_df.index.year.unique():
                    year_data_obs = obs_df[obs_df.index.year == year]
                    year_data_sim = sim_df[sim_df.index.year == year]
                    for month in range(1, 13):
                        month_data_obs = year_data_obs[year_data_obs.index.month == month]
                        month_data_sim = year_data_sim[year_data_sim.index.month == month]
                        if month in dmonths:
                        #     final_obs_list = pd.concat([final_obs_list] + [month_data_obs]*3)
                        #     final_sim_list = pd.concat([final_sim_list] + [month_data_sim]*3)
                        # else:
                        #     final_obs_list = pd.concat([final_obs_list, month_data_obs])
                        #     final_sim_list = pd.concat([final_sim_list, month_data_sim])
                            for _ in range(3):
                                final_obs_list.append(month_data_obs.values.flatten())
                                final_sim_list.append(month_data_sim.values.flatten())
                        else:
                            final_obs_list.append(month_data_obs.values.flatten())
                            final_sim_list.append(month_data_sim.values.flatten())
                # Reset the index of the final DataFrame
                # final_obs_list.reset_index(inplace=True)
                # final_sim_list.reset_index(inplace=True)
                # evaluation = final_obs_list.values.flatten()
                # simulation = final_sim_list.values.flatten()
                evaluation = np.concatenate(final_obs_list)
                simulation = np.concatenate(final_sim_list)

                # print("Double months",len(evaluation),len(simulation))

                    
                # obs = np.array(evaluation)
                # sim = np.array(simulation)
                # pbias = ((sim.sum() - obs.sum())/obs.sum())*100

            if (self.OF =='kge') or (self.OF == None):
                if self.ROUTING_MARGIN:
                    kge_pre = kge(evaluation[1:],simulation[0:-1])
                    kge_mid = kge(evaluation,simulation)
                    kge_post = kge(evaluation[0:-1],simulation[1:])
                    like = np.max([kge_pre,kge_mid,kge_post])
                    print(f"KGE pre: {kge_pre}, KGE mid: {kge_mid}, KGE post: {kge_post}")

                else:
                    print('We are calculating the single KGE')
                    like = kge(evaluation, simulation)
                    # like = kge(np.log(evaluation), np.log(simulation))
                if self.PBIAS_THRESHOLD != None:
                    if pbias > self.PBIAS_THRESHOLD:
                        print(f"PBias is {pbias} and threshold is {self.PBIAS_THRESHOLD}")
                        like /= 10 
            else:
                # like = 1 - abs(self.OF(evaluation,simulation))
                if self.OF == 'dr':
                    like = he.dr(evaluation,simulation)
                    if self.PBIAS_THRESHOLD != None:
                        if pbias > self.PBIAS_THRESHOLD:
                            print(f"Max yearly PBias is {pbias} and threshold is {self.PBIAS_THRESHOLD}")
                            like /= 10
                elif self.OF ==he.rmse:
                    like = -self.OF(evaluation,simulation)
                    if self.PBIAS_THRESHOLD != None:
                        if pbias > self.PBIAS_THRESHOLD:
                            print(f"Max yearly PBias is {pbias} and threshold is {self.PBIAS_THRESHOLD}")
                            like *= 10                
            if plot ==True:
                f1,ax1 = plt.subplots(figsize = (12,7))
                ax1.plot(evaluation, label = 'Observed')
                ax1.plot(simulation, label = 'Simulated')
                ax1.set_ylabel('Q [m3/s]')
                ax1.grid()
                plt.legend()
                ax1.set_title(f"{self.BASIN} {self.RUN_NAME}")
                # f1.savefig(f"/home/pwiersma/scratch/Figures/ewc_figures/{self.basin}_{self.RUN_NAME}.png",
                #             dpi = 250, bbox_inches = 'tight')
            if return_pbias and not return_filtered:
                #we want the mean not the max here 
                return like, np.mean(pbiases)
            elif return_filtered and not return_pbias:
                return like, [daterange,single_eval,single_sim]
            elif return_filtered and return_pbias:
                return like, [daterange, single_eval,single_sim], np.mean(pbiases)
            else:
                return like 











